#include <madrona/job.hpp>
#include <madrona/utils.hpp>
#include <madrona/context.hpp>

#if defined(__linux__) or defined(__APPLE__)
#include <signal.h>
#include <unistd.h>
#endif

using std::atomic_uint32_t;
using std::atomic_bool;
using std::atomic_thread_fence;
using std::memory_order;

namespace madrona {

struct alignas(64) JobQueueHead {
    atomic_uint32_t head;
    atomic_uint32_t correction;
    atomic_uint32_t auth;
};

struct alignas(8) JobQueueTail {
    atomic_uint32_t tail;
};

static_assert(alignof(JobQueueHead) % alignof(Job) == 0);
static_assert(atomic_uint32_t::is_always_lock_free);

namespace InternalConfig {
    constexpr static int jobQueueSizePerThread = 16384;
    constexpr static uint32_t jobQueueIndexMask =
        (uint32_t)jobQueueSizePerThread - 1;

    constexpr static uint32_t jobQueueSentinel = 0xFFFFFFFF;
    
    constexpr static uint32_t jobQueueBytesPerThread = []() {
        constexpr uint32_t bytes_per_thread =
            sizeof(JobQueueHead) + sizeof(JobQueueTail) + 
            jobQueueSizePerThread * sizeof(Job);

        return utils::roundUp(bytes_per_thread,
                              (uint32_t)std::alignment_of_v<JobQueueHead>);
    }();

    constexpr static uint32_t jobAllocSentinel = 0xFFFFFFFF;
    constexpr static uint32_t numJobAllocArenas = 1024;
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
        if (cur_head == InternalConfig::jobAllocSentinel) {
            FATAL("Out of job memory");
        }

        arena_idx = cur_head & 0xFFFF;
        new_head = shared.arenas[arena_idx].metadata.load(
            memory_order::relaxed);

        // Update the tag
        new_head += (1u << 16);
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
        new_head = (cur_head & 0xFFFF0000) + (1u << 16) + arena_idx;
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

    // Out of space in this arena, mark this arena as freeable
    // and get a new one
    if (new_offset + num_bytes <= arena_size_) {
        arena_offset_ = new_offset;
    } else {
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
            if (next_arena_ != InternalConfig::jobAllocSentinel) {
                cur_arena_ = next_arena_;
                next_arena_ = InternalConfig::jobAllocSentinel;
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
        if (next_arena_ == InternalConfig::jobAllocSentinel) {
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
            (i < int(num_arenas - 1)) ? i + 1 : 0xFFFFFFFF,
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

static inline void * alignQueue(void *start)
{
    return (void *)utils::roundUp((uintptr_t)start,
        (uintptr_t)std::alignment_of_v<JobQueueHead>);
}

static inline JobQueueHead * getQueueHead(
    void *queue_base, const int thread_idx)
{
    return (JobQueueHead *)((char *)queue_base +
        thread_idx * InternalConfig::jobQueueBytesPerThread);
}

static inline JobQueueTail * getQueueTail(JobQueueHead *head)
{
    return (JobQueueTail *)((char *)head + sizeof(JobQueueHead));
}

static inline Job *getJobArray(JobQueueTail *tail)
{
    return (Job *)((char *)tail + sizeof(JobQueueTail));
}

JobManager::JobManager(int desired_num_workers, int num_io,
                       Job::EntryPtr start_func, void *start_data,
                       StateManager &state_mgr, void *world_data,
                       bool pin_workers)
    : threads_(getNumWorkers(desired_num_workers) + num_io, InitAlloc()),
      alloc_state_(Alloc::makeSharedState(InitAlloc(),
                                          InternalConfig::numJobAllocArenas)),
      job_allocs_(threads_.size(), InitAlloc()),
      queue_store_(InitAlloc().alloc(
          3u * threads_.size() * InternalConfig::jobQueueBytesPerThread +
            std::alignment_of_v<JobQueueHead>)),
      high_start_(alignQueue(queue_store_)),
      normal_start_((char *)high_start_ + threads_.size() *
                  InternalConfig::jobQueueBytesPerThread),
      io_start_((char *)normal_start_ + threads_.size() *
                InternalConfig::jobQueueBytesPerThread),
      waiting_start_(nullptr), // FIXME
      io_sema_(0),
      job_counts_ {
          .numHigh = 0,
          .numOutstanding = 0,
      }
{
    for (int i = 0, n = threads_.size(); i < n; i++) {
        job_allocs_.emplace(i, alloc_state_);
    }

    auto initQueue = [](void *queue_start, int thread_idx) {
        JobQueueHead *head = getQueueHead(queue_start, thread_idx);

        new (head) JobQueueHead {
            .head = 0,
            .correction = 0,
            .auth = 0,
        };

        JobQueueTail *tail = getQueueTail(head);

        new (tail) JobQueueTail {
            .tail = 0,
        };
    };

    // Setup per-thread queues
    for (int i = 0, n = threads_.size(); i < n; i++) {
        initQueue(normal_start_, i);
        initQueue(high_start_, i);
    }

    struct StartWrapper {
        Job::EntryPtr func;
        void *data;
        atomic_bool launched;
    } start_wrapper {
        start_func,
        start_data,
        false,
    };

    // Initial job
    queueJob(0, [](Context &ctx, void *ptr) {
        auto &start = *(StartWrapper *)ptr;
        start.func(ctx, start.data);
        start.launched.store(true, memory_order::release);
    }, &start_wrapper, nullptr, 0, JobPriority::Normal);

    const int num_workers = threads_.size() - num_io;
    for (int i = 0; i < num_workers; i++) {
        threads_.emplace(i,
                         [this, i, pin_workers, &state_mgr, world_data]() {
            disableThreadSignals();
            if (pin_workers) {
                setThreadAffinity(i);
            }

            workerThread(i, state_mgr, world_data);
        });
    }

    for (int i = 0; i < num_io; i++) {
        int thread_idx = num_workers + i;
        threads_.emplace(thread_idx,
                         [this, thread_idx, &state_mgr, world_data]() {
            disableThreadSignals();
            ioThread(thread_idx, state_mgr, world_data);
        });
    }

    // Need to ensure start job has run at this point.
    // Otherwise, the start func closure (located on the stack)
    // can go out of scope before the job actually runs. This is
    // a bit of a hack to avoid needing to place the entire
    // constructor in job.inl
    while (!start_wrapper.launched.load(memory_order::acquire)) {
        workerYield();
    }
}

JobManager::~JobManager()
{
    InitAlloc().dealloc(alloc_state_.memoryBase);

    InitAlloc().dealloc(queue_store_);
}

JobID JobManager::queueJob(int thread_idx,
                           Job::EntryPtr job_func, void *job_data,
                           const JobID *deps, uint32_t num_dependencies,
                           JobPriority prio)
{
    (void)deps;
    (void)num_dependencies;

    JobQueueTail *queue_tail;
    if (prio == JobPriority::High) {
        queue_tail = getQueueTail(getQueueHead(high_start_, thread_idx));
    } else if (prio == JobPriority::Normal) {
        queue_tail = getQueueTail(getQueueHead(normal_start_, thread_idx));
    } else {
        queue_tail = getQueueTail(getQueueHead(io_start_, thread_idx));
    }

    atomic_uint32_t &tail = queue_tail->tail;

    // No one modifies queue_tail besides this thread
    uint32_t cur_tail = tail.load(memory_order::relaxed); 
    uint32_t wrapped_idx = (cur_tail & InternalConfig::jobQueueIndexMask);

    Job *job_array = getJobArray(queue_tail);

    job_array[wrapped_idx].func_ = job_func;
    job_array[wrapped_idx].data_ = job_data;

    cur_tail++;
    tail.store(cur_tail, memory_order::relaxed);

    if (prio == JobPriority::High) {
        job_counts_.numHigh.fetch_add(1, memory_order::relaxed);
    }
    if (prio == JobPriority::IO) {
        io_sema_.release(1);
    }

    job_counts_.numOutstanding.fetch_add(1, memory_order::relaxed);

    atomic_thread_fence(memory_order::release);

    return JobID(0);
}

JobID JobManager::queueJobs(int thread_idx, const Job *jobs, uint32_t num_jobs,
                           const JobID *deps, uint32_t num_dependencies,
                           JobPriority prio)
{
    (void)deps;
    (void)num_dependencies;

    JobQueueTail *queue_tail;
    if (prio == JobPriority::High) {
        queue_tail = getQueueTail(getQueueHead(high_start_, thread_idx));
    } else if (prio == JobPriority::Normal) {
        queue_tail = getQueueTail(getQueueHead(normal_start_, thread_idx));
    } else {
        queue_tail = getQueueTail(getQueueHead(io_start_, thread_idx));
    }

    atomic_uint32_t &tail = queue_tail->tail;

    // No one modifies queue_tail besides this thread
    uint32_t cur_tail = tail.load(memory_order::relaxed); 
    uint32_t wrapped_idx = (cur_tail & InternalConfig::jobQueueIndexMask);

    Job *job_array = getJobArray(queue_tail);

    uint32_t num_remaining = InternalConfig::jobQueueSizePerThread - wrapped_idx;
    uint32_t num_fit = std::min(num_remaining, num_jobs);
    memcpy(job_array + wrapped_idx, jobs, num_fit * sizeof(Job));

    if (num_remaining < num_jobs) {
        uint32_t num_wrapped = num_jobs - num_remaining;
        memcpy(job_array, jobs + num_remaining, num_wrapped * sizeof(Job));
    }

    cur_tail += num_jobs;
    tail.store(cur_tail, memory_order::relaxed);

    if (prio == JobPriority::High) {
        job_counts_.numHigh.fetch_add(num_jobs, memory_order::relaxed);
    }
    if (prio == JobPriority::IO) {
        io_sema_.release(num_jobs);
    }

    job_counts_.numOutstanding.fetch_add(num_jobs, memory_order::relaxed);

    atomic_thread_fence(memory_order::release);

    return JobID(0);
}

static inline bool checkGEWrapped(uint32_t a, uint32_t b)
{
    return a - b <= (1u << 31u);
}

static inline uint32_t getNextJobIndex(JobQueueHead *job_head,
                                       JobQueueTail *job_tail)
{
    atomic_uint32_t &head = job_head->head;
    atomic_uint32_t &correction = job_head->correction;
    atomic_uint32_t &auth = job_head->auth;
    atomic_uint32_t &tail = job_tail->tail;

    uint32_t cur_tail = tail.load(memory_order::relaxed);
    uint32_t cur_correction = correction.load(memory_order::relaxed);
    uint32_t cur_head = head.load(memory_order::relaxed);

    if (checkGEWrapped(cur_head - cur_correction, cur_tail)) {
        return InternalConfig::jobQueueSentinel;
    }

    atomic_thread_fence(memory_order::acquire);

    cur_head = head.fetch_add(1, memory_order::relaxed);
    cur_tail = tail.load(memory_order::acquire);

    if (checkGEWrapped(cur_head - cur_correction, cur_tail)) [[unlikely]] {
        correction.fetch_add(1, memory_order::release);

        return InternalConfig::jobQueueSentinel;
    }

    return auth.fetch_add(1, memory_order::acq_rel);
}

static inline bool getNextJob(void *const queue_start,
                              const int start_idx,
                              const int num_queues,
                              Job *next_job)
{
    for (int i = 0; i < num_queues; i++) {
        int queue_idx = (start_idx + i) % num_queues;
        JobQueueHead *queue_head = getQueueHead(queue_start, queue_idx);
        JobQueueTail *queue_tail = getQueueTail(queue_head);
    
        uint32_t job_idx = getNextJobIndex(queue_head, queue_tail);
        if (job_idx == InternalConfig::jobQueueSentinel) continue;
    
        uint32_t wrapped_job_idx =
            job_idx & InternalConfig::jobQueueIndexMask;
    
        // getNextJobIndex does an acquire fence so safe to read this
        *next_job = getJobArray(queue_tail)[wrapped_job_idx];
    
        // There's no protection to prevent queueJob overwriting
        // next_job
        // in between job_idx being assigned and the job actually being
        // read. If this happens it is a bug where way too many jobs are
        // being created, or jobs are being processed too slowly, so we
        // detect and crash with a fatal error (rather than silently
        // dropping or reading corrupted jobs).
    
        uint32_t post_read_tail =
            queue_tail->tail.load(memory_order::acquire);
    
        uint32_t wrapped_tail =
            post_read_tail & InternalConfig::jobQueueIndexMask;
    
        if (checkGEWrapped(post_read_tail,
            job_idx + InternalConfig::jobQueueSizePerThread)) [[unlikely]] {
            // Could improve this by printing the job that was read
            // Or by adding (optional?) detection to queueJob to find
            // the source of the issue.
            FATAL("Queue %d has overwritten readers. Detected by thread %d.\n"
                  "Job: %u, Wrapped Job: %u - Tail: %u, Wrapped Tail: %u\n",
                  queue_idx, start_idx, job_idx, wrapped_job_idx,
                  post_read_tail, wrapped_tail);
        }
    
        return true;
    }
    
    return false;
}

void JobManager::workerThread(const int thread_idx, StateManager &state_mgr,
                              void *world_data)
{
    Context thread_job_ctx(Context::Init {
        .jobMgr = this,
        .stateMgr = &state_mgr,
        .workerIdx = thread_idx,
    });

    const int num_queues = threads_.size();

    Job next_job;

    while (job_counts_.numOutstanding.load(memory_order::acquire) > 0) {
        bool job_found = false;

        if (job_counts_.numHigh.load(memory_order::relaxed) > 0) {
            job_found =
                getNextJob(high_start_, thread_idx, num_queues, &next_job);

            if (job_found) {
                job_counts_.numHigh.fetch_sub(1, memory_order::relaxed);
            }
        }

        if (!job_found) {
            job_found =
                getNextJob(normal_start_, thread_idx, num_queues, &next_job);
        }

        // All the queues are empty
        if (!job_found) {
            workerYield();
            continue;
        }

        next_job.func_(thread_job_ctx, next_job.data_);

        job_counts_.numOutstanding.fetch_sub(1, memory_order::release);
    }
}

void JobManager::ioThread(const int thread_idx, StateManager &state_mgr,
                          void *world_data)
{
    Context thread_job_ctx(Context::Init {
        .jobMgr = this,
        .stateMgr = &state_mgr,
        .workerIdx = thread_idx,
    });

    const int num_queues = threads_.size();

    Job next_job;

    while (job_counts_.numOutstanding.load(memory_order::acquire) > 0) {
        bool job_found = 
                getNextJob(io_start_, thread_idx, num_queues, &next_job);

        if (!job_found) {
            io_sema_.acquire();
            continue;
        }

        next_job.func_(thread_job_ctx, next_job.data_);

        job_counts_.numOutstanding.fetch_sub(1, memory_order::release);
    }
}

void JobManager::waitForAllFinished()
{
    for (int i = 0, n = threads_.size(); i < n; i++) {
        threads_[i].join();
    }
}

}
