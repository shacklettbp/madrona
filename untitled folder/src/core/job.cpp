/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include <madrona/job.hpp>
#include <madrona/utils.hpp>
#include <madrona/context.hpp>
#include <madrona/impl/id_map_impl.inl>

#include "worker_init.hpp"

#if defined(__linux__) or defined(__APPLE__)
#include <signal.h>
#include <unistd.h>
#endif

#include <atomic>

#if defined(MADRONA_X64)
#include <immintrin.h>
#elif defined(MADRONA_ARM)
#endif

using std::atomic_thread_fence;

namespace madrona {
namespace {

struct JobTracker {
    uint32_t parent;
    uint32_t remainingInvocations;
    uint32_t numOutstandingJobs;
};

template <typename T>
struct JobTrackerMapStore
{
    inline T & operator[](uint32_t idx);
    inline const T & operator[](uint32_t idx) const;
    JobTrackerMapStore(uint32_t) {}
    uint32_t expand(uint32_t)
    {
        FATAL("Out of job IDs\n");
    }

    static constexpr uint64_t pastMapOffset();
};

using JobTrackerMap = IDMap<JobID, JobTracker, JobTrackerMapStore>;

template <typename T>
constexpr uint64_t JobTrackerMapStore<T>::pastMapOffset()
{
    return sizeof(JobTrackerMap) - offsetof(JobTrackerMap, store_);
}

template <typename T>
T & JobTrackerMapStore<T>::operator[](uint32_t idx)
{
    return ((T *)((char *)this + pastMapOffset()))[idx];
}

template <typename T>
const T & JobTrackerMapStore<T>::operator[](uint32_t idx) const
{
    return ((const T *)((const char *)this + pastMapOffset()))[idx];
}

struct LogEntry {
    enum class Type : uint32_t {
        JobFinished,
        WaitingJobQueued,
        JobCreated,
    };

    struct JobFinished {
        JobContainerBase *jobData;
        int32_t jobIdx;
        uint32_t numCompleted;
    };

    struct WaitingJobQueued {
        void (*fnPtr)();
        JobContainerBase *jobData;
        uint32_t numInvocations;
    };

    struct JobCreated {
        uint32_t parentID;
    };

    Type type;
    union {
        JobFinished finished;
        WaitingJobQueued waiting;
        JobCreated created;
    };
};

#if 0
#include <vector>
std::vector<LogEntry> globalLog;
void printGlobalLog()
{
    for (LogEntry &entry : globalLog) {
        switch (entry.type) {
            case LogEntry::Type::JobFinished: {
                printf("F: (%u %u) %u %u\n", entry.finished.jobIdx.id, entry.finished.jobIdx.gen, entry.finished.numCompleted, entry.finished.threadID);
            } break;
            case LogEntry::Type::WaitingJobQueued: {
                printf("W: (%u %u) %u\n", entry.waiting.id.id, entry.waiting.id.gen, entry.waiting.numInvocations);
            } break;
            case LogEntry::Type::JobCreated: {
                printf("C: (%u %u) %u %u\n", entry.created.curID.id, entry.created.curID.gen, entry.created.parentID, entry.created.numInvocations);
            } break;
        }
    }
}
#endif

struct alignas(MADRONA_CACHE_LINE) WorkerState {
    uint32_t numIdleLoops;
    uint32_t numConsecutiveSchedulerCalls;
    AtomicU32 logHead; // Only modified by scheduler
    uint32_t logTailCache; // Scheduler's last read tail value
    // logTail below is only modified by worker thread.
    // Still has to be atomic for memory ordering guarantees (worker 
    // releases updates to log tail to guarantee the log entries themselves are
    // visible after an acquire fence by the scheduler
    alignas(MADRONA_CACHE_LINE) AtomicU32 logTail; 
    alignas(MADRONA_CACHE_LINE) AtomicU32 wakeUp;
};

// Plan:
// - High level goal: remove all contended atomics in job dependency system and outstanding job tracking system.
// - Centralize job system control into the "Scheduler"
// - When a worker thread can't find work, it attempts to
//   wait acquire the Scheduler lock. If this fails, another
//   thread is running the scheduler, so it will go to sleep
//   on WorkerWakeup primitive.
//      - Q: Is this going to create massive contention on the Scheduler lock, or is
//        there a way to have a relaxed load of the scheduler lock be safe / accurate (double checked locking maybe)
// - WorkerWakeup primitive: futex / WaitOnAddress / __ulock_wait + spin.
//      - Futex may not be strictly necessary here, since the assumption is likely that if we're about to wait, we're actually going to be put to sleep by the kernel. On OSX the only alternative seems to be posix cond vars though, which seem really slow. Regardless, the idea would be a couple of rounds of spinning (possibly using the PAUSE instruction?) where the worker makes sure it isn't just about to be woken up.
//          - Q: Does futex already provide this spin? A: Possibly, but it's irrelevant. This spin should actually be continually looking for new jobs after each PAUSE.
//  - Worker Logs: each worker thread gets a log. This records every job the worker has completed, including job ID and number of completed invocations.
//  - Scheduler: the scheduler has 2 jobs: 
//      - Run through all worker logs and use this to update job dependency info, moving any jobs from wait queue to run queue that have fulfilled dependencies. (Possible optimization: workers could check job dependency info and if they've fully satisified a requirement, just immediately unblock a job in order to skip needing to go through the scheduler).
//      - Wake up sleeping workers based on # of jobs that are ready for execution
//      - Related change: when worker decides to split or queue immediately runnable job, it should similarly trigger worker wakeups. Solution could be to wake up 1 worker, which immediately runs and wakes up others. Want to avoid a really heavy search over all threads each time a split occurs. Futex2 / WaitForMultipleObjects are linux / windows options as well (one thread local futex + one "Wake them all" futex).
//  - Pitfall: scheduler running is potentially a big synchronization point
//      - Option: Scheduler early outs after finding runnable N jobs
//      - Option: Some way to parallelize scheduler?
//  - MW tick synchronization:
//      - Scheduler runs: finds no runnable work, *and all other workers are sleeping*. At this point it does 3 things:
//          - Check should_exit flag: if this is true, it wakes all other workers and exits (can imagine more efficient impls here, this will cause N threads worth of scheduler entries - but whatever).sizeof(WaitQueue) +
//          - Signal external "frame done" futex.
//          - Point A: Wait on external "Launch next frame" futex. Upon wakeup, rerun scheduler
//      - JobManager updateLoop inserts dependency on JobID 0 rather than ctx.currentJobID() to update loop resubmission. The idea is that before signaling the launch next frame futex, external code updates JobID 0 (ez option, JobID 0 always has 1 outstanding job, just increase generation). This means after the futex is signaled and the worker at Point A runs, the "next frame" job will be ready and moved to the run queue by the scheduler.
//      - Alternative: no resubmission of updateJob by itself. Instead, before signalling the "launch next frame" futex, the system manually queues updateJob requests into each worker run queue and signals all workers.
//      - Issue: Entire above strategy is bugged if you have background work you want to keep going. Can not depend on workers being stopped as termination condition. Similarly, cannot guarantee the system is idle in order to slide updateJobs in manually. Need to use fake dependency strategy + an additional job dependent on ctx.currentJobID that when run increments an atomic. When that atomic reaches # worlds, we signal the "frame done" futex in order to wake up the external thread.
//          - How do we handle waking up the worker threads in this model? Update the job dependency and then check if the scheduler is locked? If not, run the scheduler ourselves from the external thread? Scheduler could have it's own run queue. Then worker thread model would be: check my run queue -> check scheduler run queue -> try stealing from all other run queues.
//  - This system almost certainly has a missed wakeup type race where the scheduler incorrectly concludes that no threads need to be woken up, just as other threads fail to acquire the scheduler lock and then go to sleep. Does this mean we need to ensure each worker thread acquires the scheduler before it sleeps?

namespace consts {
    constexpr uint64_t jobQueueStartAlignment = MADRONA_CACHE_LINE;

    constexpr int waitQueueSizePerWorld = 1024;

    constexpr int runQueueSizePerThread = 65536;
    constexpr uint32_t runQueueIndexMask = (uint32_t)runQueueSizePerThread - 1;

    template <uint64_t num_jobs>
    constexpr uint64_t computeRunQueueBytes()
    {
        static_assert(offsetof(JobManager::RunQueue, tail) ==
                      jobQueueStartAlignment);

        constexpr uint64_t bytes_per_thread =
            sizeof(JobManager::RunQueue) + num_jobs * sizeof(Job);

        return utils::roundUp(bytes_per_thread, jobQueueStartAlignment);
    }
    
    constexpr uint64_t runQueueBytesPerThread = 
        computeRunQueueBytes<runQueueSizePerThread>();

    constexpr int logSizePerThread = 4096; // FIXME: should decrease this and add functionality to force scheduler run
    constexpr int logSizeSafetyMargin = logSizePerThread >> 3;
    constexpr int logSizeMaxSafeCapacity = logSizePerThread - logSizeSafetyMargin;
    constexpr uint32_t logIndexMask = (uint32_t)logSizePerThread - 1;
    constexpr uint64_t logBytesPerThread = logSizePerThread * sizeof(LogEntry);

    constexpr uint32_t jobQueueSentinel = 0xFFFFFFFF;
    constexpr uint32_t jobAllocSentinel = 0xFFFFFFFF;
    constexpr uint32_t numJobAllocArenas = 1024;
}

inline void workerPause()
{
#if defined(MADRONA_X64)
    _mm_pause();
#elif defined(MADRONA_ARM)
#if defined(MADRONA_GCC) or defined(MADRONA_CLANG)
    asm volatile("yield");
#elif defined(MADRONA_MSVC)
    YieldProcessor();
#endif
#endif
}

inline void workerYield()
{
#if defined(__linux__) or defined(__APPLE__)
    sched_yield();
#elif defined(_WIN32)
    STATIC_UNIMPLEMENTED();
#else
    STATIC_UNIMPLEMENTED();
#endif
}

inline uint32_t acquireArena(JobManager::Alloc::SharedState &shared)
{
    uint32_t cur_head = shared.freeHead.load_acquire();
    uint32_t new_head, arena_idx;
    do {
        if (cur_head == consts::jobAllocSentinel) {
            FATAL("Out of job memory");
        }

        arena_idx = cur_head & 0xFFFF;
        new_head = shared.arenas[arena_idx].metadata.load_relaxed();

        // Update the tag
        new_head += ((uint32_t)1u << (uint32_t)16);
    } while (!shared.freeHead.compare_exchange_weak<
        sync::release, sync::acquire>(cur_head, new_head));

    // Arena metadata field is reused for counting used bytes, need to 0 out
    shared.arenas[arena_idx].metadata.store_release(0);

    return arena_idx;
}

inline void releaseArena(JobManager::Alloc::SharedState &shared,
                                uint32_t arena_idx)
{
    uint32_t cur_head = shared.freeHead.load_relaxed();
    uint32_t new_head;

    do {
        new_head = (cur_head & 0xFFFF0000) + ((uint32_t)1u << (uint32_t)16) + arena_idx;
        shared.arenas[arena_idx].metadata.store(cur_head, sync::relaxed);
    } while (!shared.freeHead.compare_exchange_weak<
        sync::release, sync::relaxed>(cur_head, new_head));
}

void disableThreadSignals()
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

int getNumWorkers(int num_workers)
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

void setThreadAffinity(int thread_idx)
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
    AtomicI32 numRemaining;
    AtomicI32 numAcked;

    inline void workerWait()
    {
        numRemaining.fetch_sub_release(1);

        while (numRemaining.load_acquire() != 0) {
            workerYield();
        }

        numAcked.fetch_add_release(1);
    }

    inline void mainWait(int num_threads)
    {
        while (numAcked.load_acquire() != num_threads) {
            workerYield();
        }
    }
};

inline WorkerState & getWorkerState(void *base, int thread_idx)
{
    return ((WorkerState *)base)[thread_idx];
}

inline LogEntry * getWorkerLog(void *base, int thread_idx)
{
    return (LogEntry *)((char *)base + consts::logBytesPerThread * thread_idx);
}

inline JobManager::RunQueue * getRunQueue(
    void *queue_base, const int thread_idx)
{
    return (JobManager::RunQueue *)((char *)queue_base +
        thread_idx * consts::runQueueBytesPerThread);
}

inline Job * getRunnableJobs(JobManager::RunQueue *queue)
{
    return (Job *)((char *)queue + sizeof(JobManager::RunQueue));
}

inline JobTrackerMap & getTrackerMap(void *base)
{
    return *(JobTrackerMap *)base;
}

inline JobTrackerMap::Cache & getTrackerCache(void *tracker_cache_base,
                                              int thread_idx)
{
    return ((JobTrackerMap::Cache *)tracker_cache_base)[thread_idx];
}

inline void decrementJobTracker(JobTrackerMap &tracker_map, 
                                JobTrackerMap::Cache &tracker_cache,
                                int32_t job_id)
{

    while (job_id != JobID::none().id) {
        JobTracker &tracker = tracker_map.getRef(job_id);

        uint32_t num_outstanding = --tracker.numOutstandingJobs;

        if (num_outstanding == 0 && tracker.remainingInvocations == 0) {
            uint32_t parent = tracker.parent;
            
            tracker_map.releaseID(tracker_cache, job_id);
#ifdef TSAN_ENABLED
            tracker_map.releaseGen(job_id);
#endif
            job_id = parent;
        } else {
            break;
        }
    }
}

inline const JobID * getJobDependencies(JobContainerBase *job_base)
{
    return (const JobID *)((char *)job_base + sizeof(JobContainerBase));
}

inline bool isRunnable(JobTrackerMap &tracker_map,
                       JobContainerBase *job_data)
{
    int num_deps = job_data->numDependencies;

    if (num_deps == 0) {
        return true;
    }

    const JobID *dependencies = getJobDependencies(job_data);
    for (int i = 0; i < num_deps; i++) {
        JobID dependency = dependencies[i];

        if (tracker_map.present(dependency)) {
            return false;
        }
    }

    return true;
}

template <typename Fn>
inline uint32_t addToRunQueueImpl(JobManager::RunQueue *run_queue,
                                  Fn &&add_cb)
{
    // No one modifies queue_tail besides this thread
    uint32_t cur_tail = run_queue->tail.load_relaxed();
    Job *job_array = getRunnableJobs(run_queue);

    uint32_t num_added = add_cb(job_array, cur_tail);

    cur_tail += num_added;
    run_queue->tail.store_release(cur_tail);

    return num_added;
}

inline void addToLog(WorkerState &worker_state, LogEntry *worker_log,
                     const LogEntry &entry)
{
    uint32_t cur_tail = worker_state.logTail.load_relaxed();
    uint32_t new_idx = cur_tail & consts::logIndexMask;

    worker_log[new_idx] = entry;

    uint32_t new_tail = cur_tail + 1;
    worker_state.logTail.store_release(new_tail);

    uint32_t log_head = worker_state.logHead.load_relaxed();
    if (new_tail - log_head >= consts::logSizePerThread) [[unlikely]] {
        for (uint32_t i = log_head; i != new_tail; i++) {
            LogEntry &debug_entry = worker_log[i & consts::logIndexMask];
            switch (debug_entry.type) {
                case LogEntry::Type::JobFinished: {
                    printf("Job finished\n");
                } break;
                case LogEntry::Type::WaitingJobQueued: {
                    printf("Waiting job queued\n");
                } break;
                case LogEntry::Type::JobCreated: {
                    printf("JobCreated\n");
                } break;
            }
        }
        FATAL("Worker filled up job system log");
    }
}

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
        uint32_t post_metadata =
            shared.arenas[cur_arena_].metadata.fetch_add_acq_rel(
                arena_used_bytes_);
        post_metadata += arena_used_bytes_;

        // Edge case, if post_metadata == 0, we can skip getting a new arena
        // because there are no active jobs left in the current arena, so
        // the cur_arena_ can immediately be reused by resetting offsets to 0
        if (post_metadata != 0) {
            // Get next free arena. First check the cached arena in next_arena_
            if (next_arena_ != consts::jobAllocSentinel) {
                cur_arena_ = next_arena_;
                next_arena_ = consts::jobAllocSentinel;
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

    uint32_t post_metadata = arena.metadata.fetch_sub_acq_rel(num_bytes);
    post_metadata -= num_bytes;

    if (post_metadata == 0) {
        // If this thread doesn't have a cached free arena, store there,
        // otherwise release to global free list
        if (next_arena_ == consts::jobAllocSentinel) {
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
            (i < int(num_arenas - 1)) ? i + 1 : consts::jobQueueSentinel,
        };
    }

    return SharedState {
        mem,
        job_mem,
        arenas,
        0,
    };
}

struct JobManager::Init {
    uint32_t numCtxUserdataBytes;
    void (*ctxInitFn)(void *, void *, WorkerInit &&);
    uint32_t numCtxBytes;
    void (*startFn)(Context *, void *);
    void *startFnData;
    void (*updateFn)(Context *, void *);
    void *updateFnData;
    int numWorkers;
    int numIO;
    int numThreads;
    StateManager *stateMgr;
    bool pinWorkers;
    void *statePtr;
    void *ctxBase;
    void *ctxUserdataBase;
    void *stateCacheBase;
    void *highBase;
    void *normalBase;
    void *ioBase;
    void *workerStateBase;
    void *logBase;
    void *waitingJobs;
    void *trackerBase;
    void *trackerCacheBase;
    int numTrackerSlots;
};

JobManager::JobManager(uint32_t num_ctx_userdata_bytes,
                       uint32_t ctx_userdata_alignment,
                       void (*ctx_init_fn)(void *, void *, WorkerInit &&),
                       uint32_t num_ctx_bytes,
                       uint32_t ctx_alignment,
                       void (*start_fn)(Context *, void *),
                       void *start_fn_data,
                       void (*update_fn)(Context *, void *),
                       void *update_fn_data,
                       int desired_num_workers,
                       int num_io,
                       StateManager *state_mgr,
                       bool pin_workers)
    : JobManager([num_ctx_userdata_bytes,
                  ctx_userdata_alignment, ctx_init_fn,
                  num_ctx_bytes, ctx_alignment,
                  start_fn, start_fn_data, update_fn, update_fn_data,
                  desired_num_workers, num_io, state_mgr, pin_workers]() {
        int num_workers = getNumWorkers(desired_num_workers);
        int num_threads = num_workers + num_io;

        uint64_t num_state_bytes = 0;

        uint64_t total_ctx_bytes =
            (uint64_t)num_threads * (uint64_t)num_ctx_bytes;
        uint64_t total_userdata_bytes = num_ctx_userdata_bytes;
#ifdef MADRONA_MW_MODE
        uint64_t num_worlds = state_mgr->numWorlds();

        total_ctx_bytes *= num_worlds;
        total_userdata_bytes *= num_worlds;
#else
        uint64_t num_worlds = 1;
#endif

        uint64_t ctx_offset = 0;
        num_state_bytes = ctx_offset + total_ctx_bytes;

        uint64_t ctx_userdata_offset = utils::roundUp(num_state_bytes,
            (uint64_t)ctx_userdata_alignment);

        num_state_bytes = ctx_userdata_offset + total_userdata_bytes;

        uint64_t state_cache_offset = utils::roundUp(num_state_bytes,
            (uint64_t)alignof(StateCache));

        num_state_bytes =
            state_cache_offset + sizeof(StateCache) * num_threads;

        uint64_t high_offset =
            utils::roundUp(num_state_bytes, consts::jobQueueStartAlignment);
        num_state_bytes =
            high_offset + num_threads * consts::runQueueBytesPerThread;

        uint64_t normal_offset =
            utils::roundUp(num_state_bytes, consts::jobQueueStartAlignment);
        num_state_bytes =
            normal_offset + num_threads * consts::runQueueBytesPerThread;

        uint64_t io_offset =
            utils::roundUp(num_state_bytes, consts::jobQueueStartAlignment);
        num_state_bytes =
            io_offset + num_threads * consts::runQueueBytesPerThread;

        uint64_t worker_state_offset = 
            utils::roundUp(num_state_bytes, (uint64_t)alignof(WorkerState));
        num_state_bytes =
            worker_state_offset + num_threads * sizeof(WorkerState);

        uint64_t log_offset =
            utils::roundUp(num_state_bytes, consts::jobQueueStartAlignment);
        num_state_bytes = log_offset +
            num_threads * consts::logSizePerThread * sizeof(LogEntry);

        uint64_t wait_offset =
            utils::roundUp(num_state_bytes, consts::jobQueueStartAlignment);
        num_state_bytes = wait_offset +
            num_worlds * consts::waitQueueSizePerWorld * sizeof(Job);

        uint64_t tracker_cache_offset =
            utils::roundUp(num_state_bytes, (uint64_t)alignof(JobTrackerMap::Cache));

        num_state_bytes =
            tracker_cache_offset + num_threads * sizeof(JobTrackerMap::Cache);

        int num_tracker_slots = num_threads * (
              consts::logSizePerThread + consts::runQueueSizePerThread);
        
        uint64_t tracker_offset =
            utils::roundUp(num_state_bytes, consts::jobQueueStartAlignment);

        static_assert(
            sizeof(JobTrackerMap) % alignof(JobTrackerMap::Node) == 0);

        num_state_bytes = tracker_offset + sizeof(JobTrackerMap) +
            num_tracker_slots * sizeof(JobTrackerMap::Node);

        // Add padding so the base pointer can be aligned
        num_state_bytes += ctx_alignment - 1;

        void *state_ptr = InitAlloc().alloc(num_state_bytes);

        char *base_ptr = (char *)utils::alignPtr(state_ptr, ctx_alignment);

        return Init {
            .numCtxUserdataBytes = num_ctx_userdata_bytes,
            .ctxInitFn = ctx_init_fn,
            .numCtxBytes = num_ctx_bytes,
            .startFn = start_fn,
            .startFnData = start_fn_data,
            .updateFn = update_fn,
            .updateFnData = update_fn_data,
            .numWorkers = num_workers,
            .numIO = num_io,
            .numThreads = num_threads,
            .stateMgr = state_mgr,
            .pinWorkers = pin_workers,
            .statePtr = state_ptr,
            .ctxBase = base_ptr + ctx_offset,
            .ctxUserdataBase = base_ptr + ctx_userdata_offset,
            .stateCacheBase = base_ptr + state_cache_offset,
            .highBase = base_ptr + high_offset,
            .normalBase = base_ptr + normal_offset,
            .ioBase = base_ptr + io_offset,
            .workerStateBase = base_ptr + worker_state_offset,
            .logBase = base_ptr + log_offset,
            .waitingJobs = base_ptr + wait_offset,
            .trackerBase = base_ptr + tracker_offset,
            .trackerCacheBase = base_ptr + tracker_cache_offset,
            .numTrackerSlots = num_tracker_slots,
        };
    }())
{}

JobManager::JobManager(const Init &init)
    : threads_(init.numThreads, InitAlloc()),
      alloc_state_(Alloc::makeSharedState(InitAlloc(),
                                          consts::numJobAllocArenas)),
      job_allocs_(threads_.size(), InitAlloc()),
      scheduler_ {
          .numWaiting = 0,
          .numSleepingWorkers = 0,
          .lock {},
      },
      state_ptr_(init.statePtr),
      high_base_(init.highBase),
      normal_base_(init.normalBase),
      io_base_(init.ioBase),
      tracker_base_(init.trackerBase),
      tracker_cache_base_(init.trackerCacheBase),
      worker_base_(init.workerStateBase),
      log_base_(init.logBase),
      waiting_jobs_(init.waitingJobs),
      num_compute_workers_(init.numWorkers),
      io_sema_(0),
      num_high_(0)
{
    for (int i = 0, n = init.numThreads; i < n; i++) {
        job_allocs_.emplace(i, alloc_state_);
    }

    auto initQueue = [](void *queue_start, int thread_idx) {
        RunQueue *queue = getRunQueue(queue_start, thread_idx);

        new (queue) RunQueue {
            .head = 0,
            .correction = 0,
            .auth = 0,
            .pad = {},
            .tail = 0,
        };
    };

    JobTrackerMap &tracker_map = getTrackerMap(tracker_base_);
    new (&tracker_map) JobTrackerMap(init.numTrackerSlots);
    
    // Setup per-thread state and queues
    for (int i = 0, n = threads_.size(); i < n; i++) {
        initQueue(normal_base_, i);
        initQueue(high_base_, i);
        initQueue(io_base_, i);

        WorkerState &worker_state = getWorkerState(worker_base_, i);
        new (&worker_state) WorkerState {
            .numIdleLoops = 0,
            .numConsecutiveSchedulerCalls = 0,
            .logHead = 0,
            .logTailCache = 0,
            .logTail = 0,
            .wakeUp = i + 1,
        };

        JobTrackerMap::Cache &cache = getTrackerCache(tracker_cache_base_, i);
        new (&cache) JobTrackerMap::Cache();
    }

    struct StartWrapper {
        void (*func)(Context *, void *);
        void *data;
        AtomicU32 remainingLaunches;
    } start_wrapper {
        init.startFn,
        init.startFnData,
#ifdef MADRONA_MW_MODE
        init.stateMgr->numWorlds(),
#else
        1,
#endif
    };

    struct StartJob : JobContainerBase {
        StartWrapper *wrapper;
    };
   
    SingleInvokeFn entry = [](Context *ctx, JobContainerBase *ptr) {
        auto &job = *(StartJob *)ptr;
        auto &start = *(job.wrapper);

        start.func(ctx, start.data);

        uint32_t job_id = ptr->id.id;
        start.remainingLaunches.fetch_sub_release(1);

        ctx->job_mgr_->markInvocationsFinished(ctx->worker_idx_, nullptr,
                                               job_id, 1);
    };

    // Initial job
    
#ifdef MADRONA_MW_MODE
    int num_worlds = init.stateMgr->numWorlds();

    HeapArray<StartJob, TmpAlloc> start_jobs(num_worlds);

    for (int i = 0; i < num_worlds; i++) {
        start_jobs[i] = StartJob {
            JobContainerBase { JobID::none(), sizeof(StartJob), (uint32_t)i,
                               0 },
            &start_wrapper,
        };
 
        queueJob(i % init.numWorkers, (void (*)())entry, &start_jobs[i], 0,
                 JobID::none().id, JobPriority::Normal);
    }
#else
    StartJob start_job {
        JobContainerBase { JobID::none(), sizeof(StartJob), 0 },
        &start_wrapper,
    };

    queueJob(0, (void (*)())entry, &start_job, 0, JobID::none().id,
             JobPriority::Normal);
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
    while (start_wrapper.remainingLaunches.load_acquire() != 0) {
        workerYield();
    }
}

JobManager::~JobManager()
{
    InitAlloc().dealloc(alloc_state_.memoryBase);

    InitAlloc().dealloc(state_ptr_);
}

JobID JobManager::getNewJobID(int thread_idx,
                              uint32_t parent_job_idx,
                              uint32_t num_invocations)
{
    JobTrackerMap &tracker_map = getTrackerMap(tracker_base_);
    JobTrackerMap::Cache &tracker_cache =
        getTrackerCache(tracker_cache_base_, thread_idx);
    WorkerState &worker_state = getWorkerState(worker_base_, thread_idx);
    LogEntry *log = getWorkerLog(log_base_, thread_idx);
    JobID new_id = tracker_map.acquireID(tracker_cache);

    JobTracker &tracker = tracker_map.getRef(new_id.id);
    tracker.parent = parent_job_idx;
    tracker.remainingInvocations = num_invocations;
    tracker.numOutstandingJobs = 1;

    addToLog(worker_state, log, LogEntry {
        .type = LogEntry::Type::JobCreated,
        .created = {
            .parentID = parent_job_idx,
        },
    });

    return new_id;
}

JobID JobManager::queueJob(int thread_idx,
                           void (*job_func)(),
                           JobContainerBase *job_data,
                           uint32_t num_invocations,
                           uint32_t parent_job_idx,
                           JobPriority prio)
{
    JobTrackerMap &tracker_map = getTrackerMap(tracker_base_);
    // num_invocations can be passed in as 0 here to signify a single
    // invocation job, but for the purposes of dependency tracking it
    // counts as a single invocation
    JobID id = getNewJobID(thread_idx, parent_job_idx, std::max(num_invocations, 1u));

    job_data->id = id;

    if (isRunnable(tracker_map, job_data)) {
        atomic_thread_fence(sync::acquire);
#ifdef TSAN_ENABLED
        {
            const JobID *dependencies = getJobDependencies(job_data);
            uint32_t num_dependencies = job_data->numDependencies;
            for (int i = 0; i < (int)num_dependencies; i++) {
                tracker_map.acquireGen(dependencies[i].id);
            }
        }
#endif
        addToRunQueue(thread_idx, prio,
            [=](Job *job_array, uint32_t cur_tail) {
                job_array[cur_tail & consts::runQueueIndexMask] = Job {
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

JobID JobManager::reserveProxyJobID(int thread_idx, uint32_t parent_job_idx)
{
    return getNewJobID(thread_idx, parent_job_idx, 1);
}

void JobManager::markInvocationsFinished(int thread_idx,
                                         JobContainerBase *job_data,
                                         int32_t job_idx,
                                         uint32_t num_invocations)
{
    WorkerState &worker_state = getWorkerState(worker_base_, thread_idx);
    LogEntry *log = getWorkerLog(log_base_, thread_idx);

    addToLog(worker_state, log, LogEntry {
        .type = LogEntry::Type::JobFinished,
        .finished = {
            .jobData = job_data,
            .jobIdx = job_idx,
            .numCompleted = num_invocations,
        },
    });
}

template <typename Fn>
void JobManager::addToRunQueue(int thread_idx,
                               JobPriority prio,
                               Fn &&add_cb)
{
    RunQueue *queue;
    if (prio == JobPriority::High) {
        queue = getRunQueue(high_base_, thread_idx);
    } else if (prio == JobPriority::Normal) {
        queue = getRunQueue(normal_base_, thread_idx);
    } else {
        queue = getRunQueue(io_base_, thread_idx);
    }

    uint32_t num_added = addToRunQueueImpl(queue, std::forward<Fn>(add_cb));

    if (prio == JobPriority::High) {
        num_high_.fetch_add_relaxed(num_added);
    }
    if (prio == JobPriority::IO) {
        io_sema_.release(num_added);
    }
}

void JobManager::addToWaitQueue(int thread_idx,
                                void (*job_func)(),
                                JobContainerBase *job_data,
                                uint32_t num_invocations,
                                JobPriority prio)
{
    // FIXME Priority is dropped on jobs that need to wait
    (void)prio;

    WorkerState &worker_state = getWorkerState(worker_base_, thread_idx);
    LogEntry *log = getWorkerLog(log_base_, thread_idx);

    addToLog(worker_state, log, LogEntry {
        .type = LogEntry::Type::WaitingJobQueued,
        .waiting = {
            job_func,
            job_data,
            num_invocations,
        },
    });
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

    AtomicU32 &tail = queue_tail->tail;

    // No one modifies queue_tail besides this thread
    uint32_t cur_tail = tail.load(sync::relaxed); 
    uint32_t wrapped_idx = (cur_tail & consts::jobQueueIndexMask);

    Job *job_array = getRunnableJobs(queue_tail);

    uint32_t num_remaining = consts::jobQueueSizePerThread - wrapped_idx;
    uint32_t num_fit = std::min(num_remaining, num_jobs);
    memcpy(job_array + wrapped_idx, jobs, num_fit * sizeof(Job));

    if (num_remaining < num_jobs) {
        uint32_t num_wrapped = num_jobs - num_remaining;
        memcpy(job_array, jobs + num_remaining, num_wrapped * sizeof(Job));
    }

    cur_tail += num_jobs;
    tail.store(cur_tail, sync::relaxed);

    if (prio == JobPriority::High) {
        num_high_.fetch_add(num_jobs, sync::relaxed);
    }
    if (prio == JobPriority::IO) {
        io_sema_.release(num_jobs);
    }

    num_outstanding_.fetch_add(num_jobs, sync::relaxed);

    atomic_thread_fence(sync::release);

    return JobID(0);
}
#endif

enum class JobManager::WorkerControl : uint64_t {
    Run,
    LoopIdle,
    LoopBusy,
    Sleep,
    Exit,
};

JobManager::WorkerControl JobManager::schedule(int thread_idx, Job *run_job)
{
    JobTrackerMap &tracker_map = getTrackerMap(tracker_base_);
    JobTrackerMap::Cache &tracker_cache =
        getTrackerCache(tracker_cache_base_, thread_idx);
    WorkerState &scheduling_worker =
        getWorkerState(worker_base_, thread_idx);
    scheduling_worker.numConsecutiveSchedulerCalls++;

    Job *waiting_jobs = (Job *)waiting_jobs_;
    CountT cur_num_waiting = CountT(scheduler_.numWaiting);

    auto handleJobFinished = [&](const LogEntry::JobFinished &finished) {
        JobTracker &tracker =
            tracker_map.getRef(finished.jobIdx);
        uint32_t remaining = tracker.remainingInvocations;
        remaining -= finished.numCompleted;
        tracker.remainingInvocations = remaining;

        if (remaining == 0) {
            if (finished.jobData != nullptr) {
                deallocJob(thread_idx, finished.jobData,
                           finished.jobData->jobSize);
            }

            decrementJobTracker(tracker_map, tracker_cache,
                                finished.jobIdx);
        }
    };

    auto handleWaitingQueued = [&](const LogEntry::WaitingJobQueued &waiting) {
        waiting_jobs[cur_num_waiting++] = Job {
            waiting.fnPtr,
            waiting.jobData,
            0,
            waiting.numInvocations,
        };
    };

    auto handleJobCreated = [&](const LogEntry::JobCreated &created) {
        uint32_t parent_id = created.parentID;
        if (parent_id != ~0u) {
            JobTracker &parent_tracker = tracker_map.getRef(parent_id);
            parent_tracker.numOutstandingJobs++;
        }
    };

    // First, read all the log tails and cache them. This allows us to do
    // a single acquire release barrier (dmb on arm) to ensure that log entries
    // are consistent with job tails, as well as to release JobTracker
    // generation updates in bulk.

    for (int64_t i = 0, n = threads_.size(); i < n; i++) { 
        WorkerState &worker_state = getWorkerState(worker_base_, i);
        worker_state.logTailCache = worker_state.logTail.load_relaxed();
        TSAN_ACQUIRE(&worker_state.logTail);
    }

    // Release half synchronizes all the releaseID calls under handleJobFinished
    // to ensure that when isRunnable is called outside the scheduler, the
    // job skipping the waitlist is synchronized-with the thread that finished
    // the dependencies.
    atomic_thread_fence(sync::acq_rel);

    // First, we read all the logs.
    for (int64_t i = 0, n = threads_.size(); i != n; i++) {
        int64_t offset = i + thread_idx;
        int64_t worker_idx = offset < n ? offset : offset - n;
        WorkerState &worker_state = getWorkerState(worker_base_, worker_idx);
        LogEntry *log = getWorkerLog(log_base_, worker_idx);

        uint32_t log_tail = worker_state.logTailCache;
        uint32_t log_head = worker_state.logHead.load_relaxed();

        for (; log_head != log_tail; log_head++) {
            LogEntry &entry = log[log_head & consts::logIndexMask];

            switch (entry.type) {
                case LogEntry::Type::JobFinished: {
                    handleJobFinished(entry.finished);
                } break;
                case LogEntry::Type::WaitingJobQueued: {
                    handleWaitingQueued(entry.waiting);
                } break;
                case LogEntry::Type::JobCreated: {
                    handleJobCreated(entry.created);
                } break;
            }
        }

        worker_state.logHead.store(log_head, sync::relaxed);
    }

    // Move all now runnable jobs to the scheduler's global run queue
    
    RunQueue *sched_run = getRunQueue(normal_base_, thread_idx);

    Job *sched_run_jobs = getRunnableJobs(sched_run);
    uint32_t cur_run_tail = sched_run->tail.load_relaxed();
    int64_t num_new_invocations = 0;
    int64_t compaction_offset = 0;

    bool first_found_job = true;
    for (int64_t i = 0; i < cur_num_waiting; i++) {
        Job &job = waiting_jobs[i];
        if (isRunnable(tracker_map, job.data)) {
            uint32_t num_invocations = job.numInvocations;

            // num_invocations == 0 is a special case that indicates a one-off
            // submission as opposed to a parallel for / multi invocation
            // submission. For the scheduler's purpose, this counts as one
            // invocation regardless
            num_new_invocations += num_invocations > 0 ? num_invocations : 1;

            if (first_found_job) {
                *run_job = job;
                first_found_job = false;
            } else {
                sched_run_jobs[cur_run_tail & consts::runQueueIndexMask] = job;
                cur_run_tail++;
            }
        } else {
            int64_t cur_compaction_offset = compaction_offset++;
            if (i != cur_compaction_offset) {
                waiting_jobs[cur_compaction_offset] = job;
            }
        }
    }
    scheduler_.numWaiting = compaction_offset;

    if (num_new_invocations == 0) {
        uint32_t sched_run_auth = sched_run->auth.load_relaxed();

        if (sched_run_auth == cur_run_tail) {
            if (scheduling_worker.numConsecutiveSchedulerCalls > 1) {
                if (scheduling_worker.wakeUp.load_relaxed() != 0) {
                    scheduler_.numSleepingWorkers++;
                }

                if (scheduler_.numWaiting == 0 &&
                    scheduler_.numSleepingWorkers == num_compute_workers_) {
                    for (int64_t i = 0; i < num_compute_workers_; i++) {
                        WorkerState &worker_state =
                            getWorkerState(worker_base_, i);
                        worker_state.wakeUp.store(~0_u32,
                                                  sync::relaxed);
                        worker_state.wakeUp.notify_one();
                    }
                    return WorkerControl::Exit;
                } else {
                    getWorkerState(worker_base_, thread_idx)
                        .wakeUp.store(0, sync::relaxed);
                    return WorkerControl::Sleep;
                }
            } else {
                return WorkerControl::LoopIdle;
            }
        } else {
            return WorkerControl::LoopBusy;
        }
    }
    
    sched_run->tail.store_release(cur_run_tail);

    // Wake up compute workers based on # of jobs
    int64_t num_compute_workers = num_compute_workers_;
    int64_t num_wakeup = std::min(num_compute_workers, num_new_invocations);

    for (int64_t i = 0; num_wakeup > 0 && i < num_compute_workers; i++) {
        WorkerState &worker_state = getWorkerState(worker_base_, i);

        if (worker_state.wakeUp.load_relaxed() == 0) {
            worker_state.wakeUp.store_relaxed((uint32_t)thread_idx + 1);
            worker_state.wakeUp.notify_one();

            num_wakeup--;
            scheduler_.numSleepingWorkers--;
        }
    }

    return WorkerControl::Run;
}

uint32_t JobManager::dequeueJobIndex(RunQueue *job_queue)
{
    AtomicU32 &head = job_queue->head;
    AtomicU32 &correction = job_queue->correction;
    AtomicU32 &auth = job_queue->auth;
    AtomicU32 &tail = job_queue->tail;

    uint32_t cur_tail = tail.load_relaxed();
    uint32_t cur_correction = correction.load_relaxed();
    uint32_t cur_head = head.load_relaxed();

    if (isQueueEmpty(cur_head, cur_correction, cur_tail)) {
        return consts::jobQueueSentinel;
    }

    atomic_thread_fence(sync::acquire);
    TSAN_ACQUIRE(&tail);
    TSAN_ACQUIRE(&correction);
    TSAN_ACQUIRE(&head);

    cur_head = head.fetch_add_relaxed(1);
    cur_tail = tail.load_acquire();

    if (isQueueEmpty(cur_head, cur_correction, cur_tail)) [[unlikely]] {
        correction.fetch_add_release(1);
        return consts::jobQueueSentinel;
    }

    // Note, there is some non intuitive behavior here, where the value of idx
    // can seem to be past cur_tail above. This isn't a case where too many
    // items have been dequeued, instead, the producer has added another item
    // to the queue and another consumer thread has come in and dequeued
    // the item this thread was planning on dequeuing, so this thread picks
    // up the later item. If tail is re-read after the fetch add below,
    // everything would appear consistent.
    return auth.fetch_add_acq_rel(1);
}

JobManager::WorkerControl JobManager::tryScheduling(
    JobManager::WorkerControl default_ctrl, int thread_idx, Job *next_job) {
    if (scheduler_.lock.tryLock()) {
        default_ctrl = schedule(thread_idx, next_job);
        scheduler_.lock.unlock();
    }
    return default_ctrl;
}

JobManager::WorkerControl JobManager::getNextJob(void *const queue_base,
                                                 int thread_idx,
                                                 int init_search_idx,
                                                 bool run_scheduler,
                                                 Job *next_job)
{
    WorkerControl sched_ctrl = WorkerControl::LoopIdle;

    WorkerState &worker_state = getWorkerState(worker_base_, thread_idx);
    uint32_t cur_tail = worker_state.logTail.load_relaxed();
    uint32_t log_head = worker_state.logHead.load_relaxed();
    // Determine if log capacity is too high (and we should try scheduling).
    if (cur_tail - log_head > consts::logSizeMaxSafeCapacity) {
        return tryScheduling(WorkerControl::LoopBusy, thread_idx, next_job);
    }

    // First, check the current thread's queue
    RunQueue *queue = getRunQueue(queue_base, init_search_idx);
    uint32_t job_idx = dequeueJobIndex(queue);

    if (run_scheduler && job_idx == consts::jobQueueSentinel) {
        sched_ctrl =
            tryScheduling(WorkerControl::LoopIdle, thread_idx, next_job);
        if (sched_ctrl != WorkerControl::LoopIdle) {
            return sched_ctrl;
        }
    }

    // Try work stealing
    if (job_idx == consts::jobQueueSentinel) {
        int64_t num_queues = threads_.size();
        for (int64_t i = 1; i < num_queues; i++) {
            int64_t unwrapped_idx = i + thread_idx;
            int64_t queue_idx = unwrapped_idx < num_queues ?
                unwrapped_idx : unwrapped_idx - num_queues;

            queue = getRunQueue(queue_base, queue_idx);
        
            job_idx = dequeueJobIndex(queue);
            if (job_idx != consts::jobQueueSentinel) {
                break;
            }
        }
    }

    if (job_idx == consts::jobQueueSentinel) {
        return WorkerControl::LoopIdle;
    }

    *next_job = getRunnableJobs(queue)[job_idx & consts::runQueueIndexMask];

    // There's no protection to prevent queueJob overwriting next_job
    // in between job_idx being assigned and the job actually being
    // read. If this happens it is a bug where way too many jobs are
    // being created, or jobs are being processed too slowly, so we
    // detect and crash with a fatal error (rather than silently
    // dropping or reading corrupted jobs).
    
    uint32_t post_read_tail = queue->tail.load_acquire();
    
    if (post_read_tail - job_idx > consts::runQueueSizePerThread) [[unlikely]] {
        // Note, this is not ideal because it doesn't detect the source
        // of the issue. The tradeoff is that we skip needing to read
        // the head information when queueing jobs, whereas this
        // code already has to read the tail once before.
        FATAL("Job queue has overwritten readers. Detected by thread %d.\n"
              "Job: %u, Tail: %u, Difference: %u, Queue: %p\n",
              thread_idx, job_idx, post_read_tail, post_read_tail - job_idx,
              queue);
    }

    return WorkerControl::Run;
}

void JobManager::splitJob(MultiInvokeFn fn_ptr, JobContainerBase *job_data,
                          uint32_t invocation_offset, uint32_t num_invocations,
                          RunQueue *run_queue)
{
    void (*generic_fn)() = (void (*)())fn_ptr;
    if (num_invocations == 1) {
        addToRunQueueImpl(run_queue,
            [=](Job *job_array, uint32_t cur_tail) {
                job_array[cur_tail & consts::runQueueIndexMask] = Job {
                    .func = generic_fn,
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
        addToRunQueueImpl(run_queue,
            [=](Job *job_array, uint32_t cur_tail) {
                uint32_t first_idx =
                    cur_tail & consts::runQueueIndexMask;
    
                uint32_t second_idx =
                    (cur_tail + 1) & consts::runQueueIndexMask;
    
                job_array[first_idx] = Job {
                    .func = generic_fn,
                    .data = job_data,
                    .invocationOffset = a_offset,
                    .numInvocations = a_num_invocations,
                };
    
                job_array[second_idx] = Job {
                    .func = generic_fn,
                    .data = job_data,
                    .invocationOffset = b_offset,
                    .numInvocations = b_num_invocations,
                };
    
                return 2u;
            });
    }
}

void JobManager::runJob(const int thread_idx,
                        Context *ctx,
                        void (*generic_fn)(),
                        JobContainerBase *job_data,
                        uint32_t invocation_offset,
                        uint32_t num_invocations)
{
    ctx->cur_job_id_ = job_data->id;

    if (num_invocations == 0) {
        auto fn = (SingleInvokeFn)generic_fn;
        fn(ctx, job_data);
        return;
    } else {
        // FIXME, figure out relationship between different queue priorities
        // Should the normal priority queue always be the work indicator here?
        RunQueue *check_queue = getRunQueue(normal_base_, thread_idx);

        auto fn = (MultiInvokeFn)generic_fn;
        fn(ctx, job_data, invocation_offset, num_invocations, check_queue);
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

    WorkerState &worker_state =
        getWorkerState(worker_base_, thread_idx);

    auto runCurJob = [&]() MADRONA_ALWAYS_INLINE {
        worker_state.numConsecutiveSchedulerCalls = 0;
    
#ifdef MADRONA_MW_MODE
        Context *ctx = (Context *)((char *)context_base + 
            (uint64_t)cur_job.data->worldID * (uint64_t)num_context_bytes);
#endif

        runJob(thread_idx, ctx, cur_job.func, cur_job.data,
               cur_job.invocationOffset, cur_job.numInvocations);
    };

    while (true) {
        WorkerControl worker_ctrl = WorkerControl::LoopIdle;
        if (num_high_.load_relaxed() > 0) {
            worker_ctrl = getNextJob(high_base_, thread_idx, thread_idx,
                                     false, &cur_job);

            if (worker_ctrl == WorkerControl::Run) {
                num_high_.fetch_sub_relaxed(1);
            }
        } 

        if (worker_ctrl != WorkerControl::Run) [[likely]] {
            worker_ctrl = getNextJob(normal_base_, thread_idx, thread_idx,
                                     true, &cur_job);
        }

        if (worker_ctrl == WorkerControl::Run) {
            runCurJob();
        } else if (worker_ctrl == WorkerControl::LoopIdle) {
            // No available work and couldn't run scheduler
            workerPause();
            worker_state.numIdleLoops++;
        } else if (worker_ctrl == WorkerControl::LoopBusy) {
            continue;
        } else if (worker_ctrl == WorkerControl::Sleep) [[unlikely]] {
            worker_state.wakeUp.wait<sync::relaxed>(0);
            uint32_t wakeup_idx = worker_state.wakeUp.load_relaxed();
            if (wakeup_idx == ~0_u32) [[unlikely]] {
                break;
            }

            int wakeup_search_idx = (int)wakeup_idx - 1;

            worker_ctrl = getNextJob(normal_base_, thread_idx,
                wakeup_search_idx, false, &cur_job);

            if (worker_ctrl == WorkerControl::Run) {
                runCurJob();
            }
        } else if (worker_ctrl == WorkerControl::Exit) [[unlikely]] {
            break;
        }
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
        WorkerControl worker_ctrl = getNextJob(io_base_, thread_idx,
                                               thread_idx, false, &cur_job);

        if (worker_ctrl != WorkerControl::Run) {
            io_sema_.acquire();
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
