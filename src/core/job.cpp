#include <madrona/job.hpp>
#include <madrona/utils.hpp>
#include <madrona/context.hpp>

#include "worker_init.hpp"

#if defined(__linux__) or defined(__APPLE__)
#include <signal.h>
#include <unistd.h>
#endif

#include <atomic>

using std::atomic_uint32_t;
using std::atomic_bool;
using std::atomic_thread_fence;
using std::memory_order;

namespace madrona {

struct JobQueue {
    atomic_uint32_t head;
    atomic_uint32_t correction;
    atomic_uint32_t auth;
    char pad[116];
    atomic_uint32_t tail;
};

struct WaitQueue {
    utils::SpinLock lock;
    uint32_t head;
    uint32_t tail;
};

static_assert(atomic_uint32_t::is_always_lock_free);

struct JobTrackerInfo {
    // This alignment requirement is necessary to get
    // efficient 64 bit atomics out of clang with libstdcxx on linux
    struct alignas(std::atomic_uint64_t) Head {
        uint32_t idx;
        uint32_t gen;
    };

    std::atomic<Head> head;
    static_assert(decltype(head)::is_always_lock_free);
};

struct JobTracker {
    atomic_uint32_t gen;
    atomic_uint32_t parent;
    atomic_uint32_t numOutstanding;
    atomic_uint32_t remainingInvocations;
};

namespace ICfg {
    constexpr static int waitQueueSizePerThread = 1024;
    constexpr static int runQueueSizePerThread = 4096;

    constexpr static uint32_t waitQueueIndexMask =
        (uint32_t)waitQueueSizePerThread - 1;
    constexpr static uint32_t runQueueIndexMask =
        (uint32_t)runQueueSizePerThread - 1;

    constexpr static uint32_t jobQueueSentinel = 0xFFFFFFFF;

    constexpr static uint64_t jobQueueStartAlignment = 128;

    template <uint64_t num_jobs>
    static constexpr uint64_t computeWaitQueueBytes()
    {
        constexpr uint64_t bytes_per_thread =
            sizeof(WaitQueue) + num_jobs * sizeof(Job);

        return utils::roundUp(bytes_per_thread, jobQueueStartAlignment);
    }

    constexpr static uint64_t waitQueueBytesPerThread =
        computeWaitQueueBytes<waitQueueSizePerThread>();

    template <uint64_t num_jobs>
    static constexpr uint64_t computeRunQueueBytes()
    {
        static_assert(offsetof(JobQueue, tail) == jobQueueStartAlignment);

        constexpr uint64_t bytes_per_thread =
            sizeof(JobQueue) + num_jobs * sizeof(Job);

        return utils::roundUp(bytes_per_thread, jobQueueStartAlignment);
    }
    
    constexpr static uint64_t runQueueBytesPerThread =
        computeRunQueueBytes<runQueueSizePerThread>();

    constexpr static uint32_t jobAllocSentinel = 0xFFFFFFFF;
    constexpr static uint32_t numJobAllocArenas = 1024;

    constexpr static uint32_t jobTrackerTerm = ~0u;
}

static inline bool checkGEWrapped(uint32_t a, uint32_t b)
{
    return a - b <= (1u << 31u);
}

static inline void workerYield()
{
#if defined(__linux__) or defined(__APPLE__)
    sched_yield();
#elif defined(_WIN32)
    STATIC_UNIMPLEMENTED();
#else
    STATIC_UNIMPLEMENTED();
#endif
}

static inline uint32_t acquireArena(JobManager::Alloc::SharedState &shared)
{
    uint32_t cur_head = shared.freeHead.load(memory_order::acquire);
    uint32_t new_head, arena_idx;
    do {
        if (cur_head == ICfg::jobAllocSentinel) {
            FATAL("Out of job memory");
        }

        arena_idx = cur_head & 0xFFFF;
        new_head = shared.arenas[arena_idx].metadata.load(
            memory_order::relaxed);

        // Update the tag
        new_head += ((uint32_t)1u << (uint32_t)16);
    } while (!shared.freeHead.compare_exchange_weak(cur_head, new_head,
        memory_order::release, memory_order::acquire));

    // Arena metadata field is reused for counting used bytes, need to 0 out
    shared.arenas[arena_idx].metadata.store(0, memory_order::release);

    return arena_idx;
}

static inline void releaseArena(JobManager::Alloc::SharedState &shared,
                                uint32_t arena_idx)
{
    uint32_t cur_head = shared.freeHead.load(memory_order::relaxed);
    uint32_t new_head;

    do {
        new_head = (cur_head & 0xFFFF0000) + ((uint32_t)1u << (uint32_t)16) + arena_idx;
        shared.arenas[arena_idx].metadata.store(cur_head, memory_order::relaxed);
    } while (!shared.freeHead.compare_exchange_weak(cur_head, new_head,
        memory_order::release, memory_order::relaxed));
}

JobManager::Alloc::Alloc(SharedState &shared)
    : cur_arena_(acquireArena(shared)),
      next_arena_(acquireArena(shared)),
      arena_offset_(0),
      arena_used_bytes_(0)
{}

void * JobManager::Alloc::alloc(SharedState &shared,
                                uint32_t num_bytes,
                                uint32_t alignment)
{
    // Get offset necessary to meet alignment requirements.
    // Alignment must be less than maxJobAlignment (otherwise base address not
    // guaranteed to meet alignment).
    uint32_t new_offset = utils::roundUpPow2(arena_offset_, alignment);

    if (new_offset + num_bytes <= arena_size_) {
        arena_offset_ = new_offset;
    } else {
        // Out of space in this arena, mark this arena as freeable
        // and get a new one

        // Marking the arena as freeable just involves adding the total memory
        // used in the arena to the arena's metadata value. Once all jobs in
        // the arena have been freed these values will cancel out and the
        // metadata value will be zero.
        uint32_t post_metadata = shared.arenas[cur_arena_].metadata.fetch_add(
            arena_used_bytes_, memory_order::acq_rel);
        post_metadata += arena_used_bytes_;

        // Edge case, if post_metadata == 0, we can skip getting a new arena
        // because there are no active jobs left in the current arena, so
        // the cur_arena_ can immediately be reused by resetting offsets to 0
        if (post_metadata != 0) {
            // Get next free arena. First check the cached arena in next_arena_
            if (next_arena_ != ICfg::jobAllocSentinel) {
                cur_arena_ = next_arena_;
                next_arena_ = ICfg::jobAllocSentinel;
            } else {
                cur_arena_ = acquireArena(shared);
            }
        }
        
        arena_offset_ = 0;
        arena_used_bytes_ = 0;
    }

    void *mem = 
        (char *)shared.jobMemory + arena_size_ * cur_arena_ + arena_offset_;

    arena_offset_ += num_bytes;

    // Need to track arena_used_bytes_ separately from arena_offset_,
    // because deallocation code doesn't know how many extra bytes get added
    // to each job for alignment padding reasons.
    arena_used_bytes_ += num_bytes;

    return mem;
}

void JobManager::Alloc::dealloc(SharedState &shared,
                                void *ptr, uint32_t num_bytes)
{
    size_t ptr_offset = (char *)ptr - (char *)shared.jobMemory;
    uint32_t arena_idx = ptr_offset / arena_size_;

    Arena &arena = shared.arenas[arena_idx];

    uint32_t post_metadata = arena.metadata.fetch_sub(num_bytes,
                                                      memory_order::acq_rel);
    post_metadata -= num_bytes;

    if (post_metadata == 0) {
        // If this thread doesn't have a cached free arena, store there,
        // otherwise release to global free list
        if (next_arena_ == ICfg::jobAllocSentinel) {
            next_arena_ = arena_idx;
        } else {
            releaseArena(shared, arena_idx);
        }
    }
}

JobManager::Alloc::SharedState JobManager::Alloc::makeSharedState(
    InitAlloc alloc, uint32_t num_arenas)
{
    if (num_arenas > 65536) {
        FATAL("Job allocator can only support up to 2^16 arenas.");
    }

    uint64_t total_bytes = (maxJobAlignment - 1) + num_arenas * arena_size_ +
        num_arenas * sizeof(Arena);

    void *mem = alloc.alloc(total_bytes);

    void *job_mem =
        (void *)utils::roundUp((uintptr_t)mem, (uintptr_t)maxJobAlignment);

    Arena *arenas = (Arena *)((char *)job_mem + arena_size_ * num_arenas);

    // Build initial linear freelist
    for (int i = 0; i < (int)num_arenas; i++) {
        new (&arenas[i]) Arena {
            (i < int(num_arenas - 1)) ? i + 1 : ICfg::jobQueueSentinel,
        };
    }

    return SharedState {
        mem,
        job_mem,
        arenas,
        0,
    };
}

static void disableThreadSignals()
{
#if defined(__linux__) or defined(__APPLE__)
    sigset_t mask;
    sigfillset(&mask);
    sigdelset(&mask, SIGSEGV);
    sigdelset(&mask, SIGILL);
    sigdelset(&mask, SIGBUS);
    sigdelset(&mask, SIGTRAP);
    sigdelset(&mask, SIGFPE);
    int res = pthread_sigmask(SIG_BLOCK, &mask, nullptr); 
    bool failed = res != 0;
#elif defined(_WIN32)
    STATIC_UNIMPLEMENTED();
#else
    STATIC_UNIMPLEMENTED();
#endif

    if (failed) {
        FATAL("failed to block signals for fiber executor");
    }
}

static int getNumWorkers(int num_workers)
{
    if (num_workers != 0) {
        return num_workers; 
    }

#if defined(__linux__) or defined(__APPLE__)
    int os_num_threads = sysconf(_SC_NPROCESSORS_ONLN);

    if (os_num_threads == -1) {
        FATAL("Failed to get number of concurrent threads");
    }

    return os_num_threads;
#elif defined(_WIN32)
#else
    STATIC_UNIMPLEMENTED();
#endif
}

static void setThreadAffinity(int thread_idx)
{
#if defined(__linux__)
    cpu_set_t cpuset;
    pthread_getaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

    const int max_threads = CPU_COUNT(&cpuset);

    CPU_ZERO(&cpuset);

    if (thread_idx > max_threads) [[unlikely]] {
        FATAL("Tried setting thread affinity to %d when %d is max",
              thread_idx, max_threads);
    }

    CPU_SET(thread_idx, &cpuset);

    int res = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

    if (res != 0) {
        FATAL("Failed to set thread affinity to %d", thread_idx);
    }
#elif defined(__APPLE__)
    (void)thread_idx;
    // No thread affinity on macOS / iOS :(
#elif defined(_WIN32)
    STATIC_UNIMPLEMENTED();
#else
    STATIC_UNIMPLEMENTED();
#endif
}

// 2 phases: workers decrement numRemaining (initialized to # threads)
// and then spin until it reaches 0. Next, all workers add 1 to numAcked,
// and the main thread spins until numAcked == # threads, at which point it
// knows all threads have finished initialization. Simply waiting for
// numRemaining to be 0 is insufficient, because worker threads may still
// be spinning, waiting to see that numRemaining is 0, when the ThreadPoolInit
// struct is freed
struct ThreadPoolInit {
    std::atomic_int numRemaining;
    std::atomic_int numAcked;

    inline void workerWait()
    {
        numRemaining.fetch_sub(1, memory_order::release);

        while (numRemaining.load(memory_order::acquire) != 0) {
            workerYield();
        }

        numAcked.fetch_add(1, memory_order::release);
    }

    inline void mainWait(int num_threads)
    {
        while (numAcked.load(memory_order::acquire) != num_threads) {
            workerYield();
        }
    }
};

static inline JobQueue * getRunQueue(
    void *queue_base, const int thread_idx)
{
    return (JobQueue *)((char *)queue_base +
        thread_idx * ICfg::runQueueBytesPerThread);
}

static inline WaitQueue * getWaitQueue(
    void *queue_base, const int thread_idx)
{
    return (WaitQueue *)((char *)queue_base +
        thread_idx * ICfg::waitQueueBytesPerThread);
}


static inline Job * getJobArray(JobQueue *queue)
{
    return (Job *)((char *)queue + sizeof(JobQueue));
}

static inline Job * getWaitingJobs(WaitQueue *queue)
{
    return (Job *)((char *)queue + sizeof(WaitQueue));
}

static inline JobTrackerInfo & getTrackerInfo(void *base)
{
    return *(JobTrackerInfo *)base;
}

static inline JobTracker * getTrackerArray(JobTrackerInfo &head)
{
    return (JobTracker *)((char *)&head + sizeof(JobTrackerInfo));
}

static inline uint32_t allocateJobTrackerSlot(JobTrackerInfo &tracker_info)
{
    JobTracker *trackers = getTrackerArray(tracker_info);

    // FIXME unify this lock free linked list code with GPU impl if possible
    JobTrackerInfo::Head cur_head =
        tracker_info.head.load(memory_order::acquire);

    JobTrackerInfo::Head new_head;

    do {
        if (cur_head.idx == ICfg::jobTrackerTerm) {
            break;
        }

        JobTracker &cur_tracker = trackers[cur_head.idx];

        new_head.gen = cur_head.gen + 1;
        new_head.idx = cur_tracker.parent.load(memory_order::relaxed);
    } while (!tracker_info.head.compare_exchange_weak(
        cur_head, new_head, memory_order::release,
        memory_order::acquire));

    uint32_t job_idx = cur_head.idx;

    if (job_idx == ICfg::jobTrackerTerm) [[unlikely]] {
        FATAL("Out of job ids");
    }

    return job_idx;
}

static inline void freeJobTrackerSlot(JobTrackerInfo &tracker_info,
                                      uint32_t job_id)
{
    JobTracker *trackers = getTrackerArray(tracker_info);
    JobTracker &tracker = trackers[job_id];

    JobTrackerInfo::Head new_head;
    new_head.idx = job_id;

    JobTrackerInfo::Head cur_head =
        tracker_info.head.load(memory_order::relaxed);

    do {
        new_head.gen = cur_head.gen + 1;

        tracker.parent.store(cur_head.idx, memory_order::relaxed);
    } while (!tracker_info.head.compare_exchange_weak(
           cur_head, new_head,
           memory_order::release, memory_order::relaxed));
}

static JobID getNewJobID(JobTrackerInfo &tracker_info,
                         uint32_t parent_job_idx,
                         uint32_t num_invocations,
                         uint32_t num_outstanding)
{
    JobTracker *trackers = getTrackerArray(tracker_info);

    if (parent_job_idx != ICfg::jobTrackerTerm) {
        JobTracker &parent = trackers[parent_job_idx];
        parent.numOutstanding.fetch_add(1, memory_order::release);
    }

    uint32_t tracker_slot = allocateJobTrackerSlot(tracker_info);

    JobTracker &tracker = trackers[tracker_slot];

    // Update generation first, in case there are dependencies of the
    // parent that may still check this tracker
    uint32_t prev_gen = tracker.gen.fetch_add(1, memory_order::acq_rel);

    tracker.parent.store(parent_job_idx, memory_order::relaxed);
    tracker.numOutstanding.store(num_outstanding, memory_order::release);
    tracker.remainingInvocations.store(num_invocations, memory_order::release);
    return JobID {
        .idx = tracker_slot,
        .gen = prev_gen + 1,
    };
}

static inline void decrementJobTracker(JobTrackerInfo &tracker_info, 
                                       uint32_t job_id)
{
    JobTracker *trackers = getTrackerArray(tracker_info);

    while (job_id != ICfg::jobTrackerTerm) {
        JobTracker &tracker = trackers[job_id];

        uint32_t prev_outstanding =
            tracker.numOutstanding.fetch_sub(1, memory_order::acq_rel);

        if (prev_outstanding == 1) {
            // Parent read is synchronized with the numOutstanding modification
            uint32_t parent = tracker.parent.load(memory_order::relaxed);
            freeJobTrackerSlot(tracker_info, job_id);
            job_id = parent;
        } else {
            break;
        }
    }
}

static inline bool isRunnable(JobTrackerInfo &tracker_info,
                              JobContainerBase *job_data)
{
    const JobTracker *trackers = getTrackerArray(tracker_info);

    int num_deps = job_data->numDependencies;

    if (num_deps == 0) {
        return true;
    }

    auto dependencies = (const JobID *)(
        (char *)job_data + sizeof(JobContainerBase));

    for (int i = 0; i < num_deps; i++) {
        JobID dependency = dependencies[i];
        const JobTracker &tracker = trackers[dependency.idx];

        if (tracker.gen.load(memory_order::relaxed) != dependency.gen) {
            continue;
        }

        if (tracker.numOutstanding.load(memory_order::relaxed) > 0) {
            return false;
        }
    }

    atomic_thread_fence(memory_order::acquire);

    return true;
}

struct JobManager::Init {
    uint32_t numCtxUserdataBytes;
    void (*ctxInitFn)(void *, void *, WorkerInit &&);
    uint32_t numCtxBytes;
    void (*startFn)(Context *, void *);
    void *startData;
    int numWorkers;
    int numIO;
    int numThreads;
    StateManager *stateMgr;
    bool pinWorkers;
    void *perThreadData;
    void *ctxBase;
    void *ctxUserdataBase;
    void *stateCacheBase;
    void *highBase;
    void *normalBase;
    void *ioBase;
    void *waitingBase;
    void *trackerBase;
    int numTrackerSlots;
};

JobManager::JobManager(uint32_t num_ctx_userdata_bytes,
                       uint32_t ctx_userdata_alignment,
                       void (*ctx_init_fn)(void *, void *, WorkerInit &&),
                       uint32_t num_ctx_bytes,
                       uint32_t ctx_alignment,
                       void (*start_fn)(Context *, void *),
                       void *start_data,
                       int desired_num_workers,
                       int num_io,
                       StateManager *state_mgr,
                       bool pin_workers)
    : JobManager([num_ctx_userdata_bytes,
                  ctx_userdata_alignment, ctx_init_fn,
                  num_ctx_bytes, ctx_alignment,
                  start_fn, start_data, desired_num_workers,
                  num_io, state_mgr, pin_workers]() {
        int num_workers = getNumWorkers(desired_num_workers);
        int num_threads = num_workers + num_io;

        uint64_t num_per_thread_bytes = 0;

        uint64_t total_ctx_bytes =
            (uint64_t)num_threads * (uint64_t)num_ctx_bytes;
        uint64_t total_userdata_bytes = num_ctx_userdata_bytes;
#ifdef MADRONA_MW_MODE
        int num_worlds = state_mgr->numWorlds();

        total_ctx_bytes *= num_worlds;
        total_userdata_bytes *= num_worlds;
#endif

        uint64_t ctx_offset = 0;
        num_per_thread_bytes = ctx_offset + total_ctx_bytes;

        uint64_t ctx_userdata_offset = utils::roundUp(num_per_thread_bytes,
            (uint64_t)ctx_userdata_alignment);

        num_per_thread_bytes = ctx_userdata_offset + total_userdata_bytes;

        uint64_t state_cache_offset = utils::roundUp(num_per_thread_bytes,
            (uint64_t)alignof(StateCache));

        num_per_thread_bytes =
            state_cache_offset + sizeof(StateCache) * num_threads;

        uint64_t high_offset =
            utils::roundUp(num_per_thread_bytes, ICfg::jobQueueStartAlignment);
        num_per_thread_bytes =
            high_offset + num_threads * ICfg::runQueueBytesPerThread;

        uint64_t normal_offset =
            utils::roundUp(num_per_thread_bytes, ICfg::jobQueueStartAlignment);
        num_per_thread_bytes =
            normal_offset + num_threads * ICfg::runQueueBytesPerThread;

        uint64_t io_offset =
            utils::roundUp(num_per_thread_bytes, ICfg::jobQueueStartAlignment);
        num_per_thread_bytes =
            io_offset + num_threads * ICfg::runQueueBytesPerThread;

        uint64_t wait_offset =
            utils::roundUp(num_per_thread_bytes, ICfg::jobQueueStartAlignment);
        num_per_thread_bytes =
            wait_offset + num_threads * ICfg::waitQueueBytesPerThread;

        int num_tracker_slots = num_threads * (
              ICfg::waitQueueSizePerThread + ICfg::runQueueSizePerThread);
        uint64_t tracker_offset =
            utils::roundUp(num_per_thread_bytes, ICfg::jobQueueStartAlignment);

        static_assert(sizeof(JobTrackerInfo) % alignof(JobTracker) == 0);

        num_per_thread_bytes = tracker_offset + sizeof(JobTrackerInfo) +
            num_tracker_slots * sizeof(JobTracker);

        // Add padding so the base pointer can be aligned
        num_per_thread_bytes += ctx_alignment - 1;

        void *per_thread_data = InitAlloc().alloc(num_per_thread_bytes);

        char *base_ptr = (char *)utils::roundUp((uintptr_t)per_thread_data,
                                                (uintptr_t)ctx_alignment);

        return Init {
            .numCtxUserdataBytes = num_ctx_userdata_bytes,
            .ctxInitFn = ctx_init_fn,
            .numCtxBytes = num_ctx_bytes,
            .startFn = start_fn,
            .startData = start_data,
            .numWorkers = num_workers,
            .numIO = num_io,
            .numThreads = num_threads,
            .stateMgr = state_mgr,
            .pinWorkers = pin_workers,
            .perThreadData = per_thread_data,
            .ctxBase = base_ptr + ctx_offset,
            .ctxUserdataBase = base_ptr + ctx_userdata_offset,
            .stateCacheBase = base_ptr + state_cache_offset,
            .highBase = base_ptr + high_offset,
            .normalBase = base_ptr + normal_offset,
            .ioBase = base_ptr + io_offset,
            .waitingBase = base_ptr + wait_offset,
            .trackerBase = base_ptr + tracker_offset,
            .numTrackerSlots = num_tracker_slots,
        };
    }())
{}

JobManager::JobManager(const Init &init)
    : threads_(init.numThreads, InitAlloc()),
      alloc_state_(Alloc::makeSharedState(InitAlloc(),
                                          ICfg::numJobAllocArenas)),
      job_allocs_(threads_.size(), InitAlloc()),
      per_thread_data_(init.perThreadData),
      high_base_(init.highBase),
      normal_base_(init.normalBase),
      io_base_(init.ioBase),
      waiting_base_(init.waitingBase),
      tracker_base_(init.trackerBase),
      io_sema_(0),
      num_high_(0),
      num_outstanding_(0)
{
    for (int i = 0, n = init.numThreads; i < n; i++) {
        job_allocs_.emplace(i, alloc_state_);
    }

    auto initQueue = [](void *queue_start, int thread_idx) {
        JobQueue *queue = getRunQueue(queue_start, thread_idx);

        new (queue) JobQueue {
            .head = 0,
            .correction = 0,
            .auth = 0,
            .pad = {},
            .tail = 0,
        };
    };

    // Setup per-thread queues
    for (int i = 0, n = threads_.size(); i < n; i++) {
        initQueue(normal_base_, i);
        initQueue(high_base_, i);
        initQueue(io_base_, i);

        WaitQueue *wait_queue = getWaitQueue(waiting_base_, i);
        new (wait_queue) WaitQueue {
            {},
            0,
            0,
        };
    }

    JobTrackerInfo &tracker_info = getTrackerInfo(tracker_base_);
    tracker_info.head = {
        .idx = 0,
        .gen = 0,
    };

    JobTracker *trackers = getTrackerArray(tracker_info);

    for (int i = 0, n = init.numTrackerSlots; i < n; i++) {
        JobTracker &tracker = trackers[i];

        tracker.gen.store(0, memory_order::relaxed);
        if (i < n - 1) {
            tracker.parent.store(i + 1, memory_order::relaxed);
        } else {
            tracker.parent.store(ICfg::jobTrackerTerm, memory_order::relaxed);
        }

        tracker.numOutstanding.store(0, memory_order::relaxed);
    }

    struct StartWrapper {
        void (*func)(Context *, void *);
        void *data;
        atomic_uint32_t remainingLaunches;
    } start_wrapper {
        init.startFn,
        init.startData,
#ifdef MADRONA_MW_MODE
        init.stateMgr->numWorlds(),
#else
        1,
#endif
    };

    struct StartJob : JobContainerBase {
        StartWrapper *wrapper;
    };
   
    auto entry = [](Context *ctx, JobContainerBase *ptr, uint32_t) {
        auto &job = *(StartJob *)ptr;
        auto &start = *(job.wrapper);

        start.func(ctx, start.data);
        ctx->job_mgr_->jobFinished(job.id.idx);

        start.remainingLaunches.fetch_sub(1, memory_order::release);
    };

    // Initial job
    
#ifdef MADRONA_MW_MODE
    int num_worlds = init.stateMgr->numWorlds();

    HeapArray<StartJob, TmpAlloc> start_jobs(num_worlds);

    for (int i = 0; i < num_worlds; i++) {
        start_jobs[i] = StartJob {
            JobContainerBase { JobID::none(), (uint32_t)i, 0 },
            &start_wrapper,
        };
 
        queueJob(i % init.numWorkers, entry, &start_jobs[i], 1,
                 JobID::none().idx, JobPriority::Normal);
    }
#else
    StartJob start_job {
        JobContainerBase { JobID::none(), 0 },
        &start_wrapper,
    };

    queueJob(0, entry, &start_job, 1, JobID::none().idx, JobPriority::Normal);
#endif

    ThreadPoolInit pool_init { init.numThreads, 0 };

    for (int thread_idx = 0; thread_idx < init.numThreads; thread_idx++) {
        // Find the proper state cache for this thread and initialize it before
        // passing to context
        StateCache *thread_state_cache = (StateCache *)(
            (char *)init.stateCacheBase + thread_idx * sizeof(StateCache));
        new (thread_state_cache) StateCache();

#ifdef MADRONA_MW_MODE
        void *ctx_store = (char *)init.ctxBase + (uint64_t)thread_idx *
            (uint64_t)init.numCtxBytes * (uint64_t)num_worlds;

        for (int world_idx = 0; world_idx < num_worlds; world_idx++) {
            void *cur_ctx =
                (char *)ctx_store + world_idx * (uint64_t)init.numCtxBytes;

            void *cur_userdata = (char *)init.ctxUserdataBase +
                world_idx * (uint64_t)init.numCtxUserdataBytes;

            init.ctxInitFn(cur_ctx, cur_userdata, WorkerInit {
                .jobMgr = this,
                .stateMgr = init.stateMgr,
                .stateCache = thread_state_cache,
                .workerIdx = thread_idx,
                .worldID = (uint32_t)world_idx,
            });
        }
#else
        void *ctx_store = (char *)init.ctxBase + thread_idx * init.numCtxBytes;
        init.ctxInitFn(ctx_store, init.ctxUserdataBase, WorkerInit {
            .jobMgr = this,
            .stateMgr = init.stateMgr,
            .stateCache = thread_state_cache,
            .workerIdx = thread_idx,
        });
#endif
        threads_.emplace(thread_idx, [this](
                int thread_idx,
                void *context_base,
                uint32_t num_context_bytes,
                int num_workers,
                bool pin_workers,
                ThreadPoolInit *pool_init) {
            bool is_worker = thread_idx < num_workers;

            if (is_worker) {
                disableThreadSignals();
                if (pin_workers) {
                    setThreadAffinity(thread_idx);
                }
            }

            pool_init->workerWait();

            if (is_worker) {
                workerThread(thread_idx, context_base,
                             num_context_bytes);
            } else {
                ioThread(thread_idx, context_base,
                         num_context_bytes);
            }
        }, thread_idx, ctx_store, init.numCtxBytes, init.numWorkers,
            init.pinWorkers, &pool_init);
    }

    pool_init.mainWait(init.numThreads);

    // Need to ensure start job has run at this point.
    // Otherwise, the start function data can be freed / go out of scope
    // before the job actually runs.
    while (start_wrapper.remainingLaunches.load(memory_order::acquire) != 0) {
        workerYield();
    }
}

JobManager::~JobManager()
{
    InitAlloc().dealloc(alloc_state_.memoryBase);

    InitAlloc().dealloc(per_thread_data_);
}

JobID JobManager::queueJob(int thread_idx,
                           Job::EntryPtr job_func,
                           JobContainerBase *job_data,
                           uint32_t num_invocations,
                           uint32_t parent_job_idx,
                           JobPriority prio)
{
    JobTrackerInfo &tracker_info = getTrackerInfo(tracker_base_);

    JobID id = getNewJobID(tracker_info, parent_job_idx, num_invocations, 1);

    job_data->id = id;

    num_outstanding_.fetch_add(1, memory_order::relaxed);

    if (isRunnable(tracker_info, job_data)) {
        addToRunQueue(thread_idx, prio,
            [=](Job *job_array, uint32_t cur_tail) {
                job_array[cur_tail & ICfg::runQueueIndexMask] = Job {
                    .func = job_func,
                    .data = job_data,
                    .invocationOffset = 0,
                    .numInvocations = num_invocations,
                };

                return 1u;
            });
    } else {
        addToWaitQueue(thread_idx, job_func, job_data, num_invocations,
                       prio);
    }

    return id;
}

JobID JobManager::reserveProxyJobID(uint32_t parent_job_idx)
{
    return getNewJobID(getTrackerInfo(tracker_base_), parent_job_idx, 0, 1);
}

void JobManager::relinquishProxyJobID(uint32_t job_idx)
{
    JobTrackerInfo &tracker_info = getTrackerInfo(tracker_base_);
    decrementJobTracker(tracker_info, job_idx);
}

void JobManager::jobFinished(uint32_t job_idx)
{
    JobTrackerInfo &tracker_info = getTrackerInfo(tracker_base_);

    num_outstanding_.fetch_sub(1, memory_order::release);

    decrementJobTracker(tracker_info, job_idx);
}

bool JobManager::markInvocationFinished(JobContainerBase *job)
{
    JobTrackerInfo &tracker_info = getTrackerInfo(tracker_base_);
    JobTracker *trackers = getTrackerArray(tracker_info);

    uint32_t job_idx = job->id.idx;

    uint32_t prev_remaining_invocations =
        trackers[job_idx].remainingInvocations.fetch_sub(
            1, memory_order::acq_rel);

    bool finished = prev_remaining_invocations == 1;

    if (finished) {
        jobFinished(job_idx);
    }

    return finished;
}

template <typename Fn>
void JobManager::addToRunQueue(int thread_idx,
                               JobPriority prio,
                               Fn &&add_cb)
{
    JobQueue *queue;
    if (prio == JobPriority::High) {
        queue = getRunQueue(high_base_, thread_idx);
    } else if (prio == JobPriority::Normal) {
        queue = getRunQueue(normal_base_, thread_idx);
    } else {
        queue = getRunQueue(io_base_, thread_idx);
    }

    // No one modifies queue_tail besides this thread
    uint32_t cur_tail = queue->tail.load(memory_order::relaxed);
    Job *job_array = getJobArray(queue);

    uint32_t num_added = add_cb(job_array, cur_tail);

    if (prio == JobPriority::High) {
        num_high_.fetch_add(num_added, memory_order::relaxed);
    }
    if (prio == JobPriority::IO) {
        io_sema_.release(num_added);
    }

    cur_tail += num_added;
    queue->tail.store(cur_tail, memory_order::release);
}

void JobManager::addToWaitQueue(int thread_idx,
                                Job::EntryPtr job_func,
                                JobContainerBase *job_data,
                                uint32_t num_invocations,
                                JobPriority prio)
{
    // FIXME Priority is dropped on jobs that need to wait
    (void)prio;

    auto addJob = [&](WaitQueue *wait_queue) {
        Job *waiting_jobs = getWaitingJobs(wait_queue);

        uint32_t new_idx = ((wait_queue->tail++) & ICfg::waitQueueIndexMask);

        waiting_jobs[new_idx] = Job {
            job_func,
            job_data,
            0,
            num_invocations,
        };
    };

    // First see if there is a queue we can grab with no contention
    for (int i = 0, n = threads_.size(); i < n; i++) {
        int queue_idx = (i + thread_idx) % n;
        WaitQueue *wait_queue = getWaitQueue(waiting_base_, queue_idx);

        if (wait_queue->lock.lockNoSpin()) {
            addJob(wait_queue);
            wait_queue->lock.unlock();
            return;
        }
    }

    // Failsafe, if couldn't find queue to wait on just spin on current
    // thread's queue
    WaitQueue *default_queue = getWaitQueue(waiting_base_, thread_idx);
    default_queue->lock.lock();
    addJob(default_queue);
    default_queue->lock.unlock();
}

#if 0
JobID JobManager::queueJobs(int thread_idx, const Job *jobs, uint32_t num_jobs,
                           const JobID *deps, uint32_t num_dependencies,
                           JobPriority prio)
{
    (void)deps;
    (void)num_dependencies;

    JobQueueTail *queue_tail;
    if (prio == JobPriority::High) {
        queue_tail = getQueueTail(getQueueHead(high_base_, thread_idx));
    } else if (prio == JobPriority::Normal) {
        queue_tail = getQueueTail(getQueueHead(normal_base_, thread_idx));
    } else {
        queue_tail = getQueueTail(getQueueHead(io_base_, thread_idx));
    }

    atomic_uint32_t &tail = queue_tail->tail;

    // No one modifies queue_tail besides this thread
    uint32_t cur_tail = tail.load(memory_order::relaxed); 
    uint32_t wrapped_idx = (cur_tail & ICfg::jobQueueIndexMask);

    Job *job_array = getJobArray(queue_tail);

    uint32_t num_remaining = ICfg::jobQueueSizePerThread - wrapped_idx;
    uint32_t num_fit = std::min(num_remaining, num_jobs);
    memcpy(job_array + wrapped_idx, jobs, num_fit * sizeof(Job));

    if (num_remaining < num_jobs) {
        uint32_t num_wrapped = num_jobs - num_remaining;
        memcpy(job_array, jobs + num_remaining, num_wrapped * sizeof(Job));
    }

    cur_tail += num_jobs;
    tail.store(cur_tail, memory_order::relaxed);

    if (prio == JobPriority::High) {
        num_high_.fetch_add(num_jobs, memory_order::relaxed);
    }
    if (prio == JobPriority::IO) {
        io_sema_.release(num_jobs);
    }

    num_outstanding_.fetch_add(num_jobs, memory_order::relaxed);

    atomic_thread_fence(memory_order::release);

    return JobID(0);
}
#endif

void JobManager::findReadyJobs(int thread_idx)
{
    JobTrackerInfo &tracker_info = getTrackerInfo(tracker_base_);

    int num_queues = threads_.size();

    int num_found = 0;
    for (int i = 0; i < num_queues; i++) {
        int queue_idx = (i + thread_idx) % num_queues;
        WaitQueue *wait_queue = getWaitQueue(waiting_base_, queue_idx);

        bool lock_succeeded = wait_queue->lock.lockNoSpin();

        if (!lock_succeeded) {
            continue;
        }

        Job *waiting_jobs = getWaitingJobs(wait_queue);

        // FIXME: this currently uses a very basic strategy to keeping
        // the array contiguous, which is just to copy all still waiting
        // jobs forward in the ring buffer. This ensures order is
        // preserved, but is a lot of needless copying. Consider switching
        // to a linked list here
        uint32_t cur_tail = wait_queue->tail;
        for (uint32_t wait_idx = wait_queue->head;
             wait_idx != cur_tail; wait_idx++) {
            
            Job job = waiting_jobs[wait_idx & ICfg::waitQueueIndexMask];
            if (isRunnable(tracker_info, job.data)) {
                addToRunQueue(thread_idx, JobPriority::Normal,
                    [&job](Job *job_array, uint32_t cur_tail) {
                        job_array[cur_tail & ICfg::runQueueIndexMask] = job;

                        return 1u;
                    });
                num_found++;
            } else {
                uint32_t move_idx =
                    ((wait_queue->tail++) & ICfg::waitQueueIndexMask);
                waiting_jobs[move_idx] = job;
            }
        }

        wait_queue->head = cur_tail;

        wait_queue->lock.unlock();

        if (num_found > 0) {
            break;
        }
    }
}

static inline uint32_t getNextJobIndex(JobQueue *job_queue)
{
    atomic_uint32_t &head = job_queue->head;
    atomic_uint32_t &correction = job_queue->correction;
    atomic_uint32_t &auth = job_queue->auth;
    atomic_uint32_t &tail = job_queue->tail;

    uint32_t cur_tail = tail.load(memory_order::relaxed);
    uint32_t cur_correction = correction.load(memory_order::relaxed);
    uint32_t cur_head = head.load(memory_order::relaxed);

    if (checkGEWrapped(cur_head - cur_correction, cur_tail)) {
        return ICfg::jobQueueSentinel;
    }

    atomic_thread_fence(memory_order::acquire);

    cur_head = head.fetch_add(1, memory_order::relaxed);
    cur_tail = tail.load(memory_order::acquire);

    if (checkGEWrapped(cur_head - cur_correction, cur_tail)) [[unlikely]] {
        correction.fetch_add(1, memory_order::release);

        return ICfg::jobQueueSentinel;
    }

    // Note, there is some non intuitive behavior here, where the value of idx
    // can seem to be past cur_tail above. This isn't a case where too many
    // items have been dequeued, instead, the producer has added another item
    // to the queue and another consumer thread has come in and dequeued
    // the item this thread was planning on dequeuing, so this thread picks
    // up the later item. If tail is re-read after the fetch add below,
    // everything would appear consistent.
    return auth.fetch_add(1, memory_order::acq_rel);
}

bool JobManager::getNextJob(void *const queue_base,
                            int thread_idx,
                            bool check_waiting,
                            Job *next_job)
{
    // First, check queue start_idx (the current thread's)
    JobQueue *queue;
    uint32_t job_idx;
    {
        queue = getRunQueue(queue_base, thread_idx);
        job_idx = getNextJobIndex(queue);

        if (job_idx == ICfg::jobQueueSentinel && check_waiting) {
            // Check for waiting jobs
            findReadyJobs(thread_idx);

            // Recheck current queue
            job_idx = getNextJobIndex(queue);
        }
    }

    int num_queues = threads_.size();
    if (job_idx == ICfg::jobQueueSentinel) {
        for (int i = 1, n = num_queues - 1; i < n; i++) {
            int queue_idx = (thread_idx + i) % num_queues;
            queue = getRunQueue(queue_base, queue_idx);
        
            job_idx = getNextJobIndex(queue);
            if (job_idx != ICfg::jobQueueSentinel) {
                break;
            }
        }
    }

    if (job_idx == ICfg::jobQueueSentinel) {
        return false;
    }
    
    *next_job = getJobArray(queue)[job_idx & ICfg::runQueueIndexMask];

    // There's no protection to prevent queueJob overwriting next_job
    // in between job_idx being assigned and the job actually being
    // read. If this happens it is a bug where way too many jobs are
    // being created, or jobs are being processed too slowly, so we
    // detect and crash with a fatal error (rather than silently
    // dropping or reading corrupted jobs).
    
    uint32_t post_read_tail = queue->tail.load(memory_order::acquire);
    
    if (post_read_tail - job_idx > ICfg::runQueueSizePerThread) [[unlikely]] {
        // Note, this is not ideal because it doesn't detect the source
        // of the issue. The tradeoff is that we skip needing to read
        // the head information when queueing jobs, whereas this
        // code already has to read the tail once before.
        FATAL("Job queue has overwritten readers. Detected by thread %d.\n"
              "Job: %u, Tail: %u, Difference: %u, Queue: %p\n",
              thread_idx, job_idx, post_read_tail, post_read_tail - job_idx,
              queue);
    }

    return true;
}

void JobManager::runJob(const int thread_idx,
                        Context *ctx,
                        Job::EntryPtr fn,
                        JobContainerBase *job_data,
                        uint32_t invocation_offset,
                        uint32_t num_invocations)
{
    ctx->cur_job_id_ = job_data->id;

    // FIXME, figure out relationship between different queue priorities
    // Should the normal priority queue always be the work indicator here?
    JobQueue *check_queue = getRunQueue(normal_base_, thread_idx);

    auto checkQueueEmpty = [check_queue] {
        uint32_t cur_tail = check_queue->tail.load(memory_order::relaxed);
        uint32_t cur_correction =
            check_queue->correction.load(memory_order::relaxed);
        uint32_t cur_head = check_queue->head.load(memory_order::relaxed);

        return checkGEWrapped(cur_head - cur_correction, cur_tail);
    };

    while (num_invocations > 0) {
        uint32_t cur_offset = invocation_offset++;
        num_invocations -= 1;

        // FIXME: improvement - check empty on first iteration,
        // after that, only check every N iterations (possibly determined
        // by # of iteration CPU cycles). There are probably some
        // heuristics here like if num_invocations >> num threads,
        // leave some iterations in this loop rather than fully splitting
        if (num_invocations > 0 && checkQueueEmpty()) {
            if (num_invocations == 1) {
                addToRunQueue(thread_idx, JobPriority::Normal,
                    [=](Job *job_array, uint32_t cur_tail) {
                        job_array[cur_tail & ICfg::runQueueIndexMask] = Job {
                            .func = fn,
                            .data = job_data,
                            .invocationOffset = invocation_offset,
                            .numInvocations = 1,
                        };

                        return 1u;
                    });
            } else {
                uint32_t b_num_invocations = num_invocations / 2;
                uint32_t a_num_invocations =
                    num_invocations - b_num_invocations;

                uint32_t a_offset = invocation_offset;
                uint32_t b_offset = a_offset + a_num_invocations;

                // FIXME, again priority issues here
                addToRunQueue(thread_idx, JobPriority::Normal,
                    [=](Job *job_array, uint32_t cur_tail) {
                        uint32_t first_idx =
                            cur_tail & ICfg::runQueueIndexMask;

                        uint32_t second_idx =
                            (cur_tail + 1) & ICfg::runQueueIndexMask;

                        job_array[first_idx] = Job {
                            .func = fn,
                            .data = job_data,
                            .invocationOffset = a_offset,
                            .numInvocations = a_num_invocations,
                        };

                        job_array[second_idx] = Job {
                            .func = fn,
                            .data = job_data,
                            .invocationOffset = b_offset,
                            .numInvocations = b_num_invocations,
                        };

                        return 2u;
                    });
            }

            num_invocations = 0;
        }

        fn(ctx, job_data, cur_offset);
    }
}

void JobManager::workerThread(
    const int thread_idx, 
    void *context_base,
    uint32_t num_context_bytes)
{
#ifndef MADRONA_MW_MODE
    (void)num_context_bytes;
    Context *ctx = (Context *)context_base;
#endif

    Job cur_job;

    while (true) {
        bool job_found = false;

        if (num_high_.load(memory_order::relaxed) > 0) {
            job_found = getNextJob(high_base_, thread_idx, false, &cur_job);

            if (job_found) {
                num_high_.fetch_sub(1, memory_order::relaxed);
            }
        }

        if (!job_found) {
            job_found = getNextJob(normal_base_, thread_idx, true, &cur_job);
        }

        // All the queues are empty
        if (!job_found) {
            if (num_outstanding_.load(memory_order::acquire) > 0) {
                workerYield();
                continue;
            } else {
                break;
            }
        }

#ifdef MADRONA_MW_MODE
        Context *ctx = (Context *)((char *)context_base + 
            (uint64_t)cur_job.data->worldID * (uint64_t)num_context_bytes);
#endif

        runJob(thread_idx, ctx, cur_job.func, cur_job.data,
               cur_job.invocationOffset, cur_job.numInvocations);
    }
}

void JobManager::ioThread(
    const int thread_idx, 
    void *context_base,
    uint32_t num_context_bytes)
{
#ifndef MADRONA_MW_MODE
    (void)num_context_bytes;
    Context *ctx = (Context *)context_base;
#endif

    Job cur_job;

    while (true) {
        bool job_found = getNextJob(io_base_, thread_idx, false, &cur_job);

        if (!job_found) {
            if (num_outstanding_.load(memory_order::acquire) > 0) {
                io_sema_.acquire();
                continue;
            } else {
                break;
            }
        }

#ifdef MADRONA_MW_MODE
        Context *ctx = (Context *)((char *)context_base + 
            (uint64_t)cur_job.data->worldID * (uint64_t)num_context_bytes);
#endif

        runJob(thread_idx, ctx, cur_job.func, cur_job.data,
               cur_job.invocationOffset, cur_job.numInvocations);
    }
}

void JobManager::waitForAllFinished()
{
    for (int i = 0, n = threads_.size(); i < n; i++) {
        threads_[i].join();
    }
}

}
