#pragma once

#include <madrona/fwd.hpp>
#include <madrona/memory.hpp>
#include <madrona/heap_array.hpp>
#include <madrona/span.hpp>

#include <atomic>
#include <array>
#include <thread>
#include <semaphore>

namespace madrona {

enum class JobPriority {
    High,
    Normal,
    IO,
};

struct JobID {
    uint32_t idx;
    uint32_t gen;

    static inline JobID none();
};

struct JobContainerBase {
    JobID id;
    uint32_t numDependencies;

    template <size_t N> struct DepsArray;
};

template <typename Fn, size_t N>
struct JobContainer : public JobContainerBase {
    [[no_unique_address]] DepsArray<N> dependencies;
    [[no_unique_address]] Fn fn;

    template <typename... DepTs>
    inline JobContainer(Fn &&fn, DepTs ...deps);
};

struct Job {
    using EntryPtr = void (*)(Context &, JobContainerBase *, uint32_t);

    EntryPtr func;
    JobContainerBase *data;
    uint32_t invocationOffset;
    uint32_t numInvocations;
};

class JobManager {
public:
    template <typename StartFn, typename DataT> struct EntryConfig;

    template <typename ContextT, typename DataT, typename StartFn>
    static EntryConfig<DataT, StartFn> makeEntry(
        DataT &&ctx_data, StartFn &&start_fn);

    template <typename DataT, typename StartFn>
    JobManager(const EntryConfig<DataT, StartFn> &entry_cfg,
               int desired_num_workers,
               int num_io,
               StateManager *state_mgr,
               bool pin_workers = true);

    ~JobManager();

    inline JobID reserveProxyJobID(JobID parent_id);
    inline void relinquishProxyJobID(JobID job_id);

    template <typename ContextT, typename Fn, typename... DepTs>
    JobID queueJob(int thread_idx, Fn &&fn, uint32_t num_invocations,
                   JobID parent_id,
                   JobPriority prio = JobPriority::Normal,
                   DepTs ...deps);

#if 0
    JobID queueJobs(int thread_idx, JobID parent_id,
                    const Job *jobs, uint32_t num_jobs,
                    const JobID *deps, uint32_t num_dependencies,
                    JobPriority prio = JobPriority::Normal);
#endif

    void waitForAllFinished();

    // Custom allocator that recycles small arenas out of a large chunk of
    // preallocated memory.
    class Alloc {
    public:
        struct Arena {
            // Doubles as next pointer in freelist or
            // count of num bytes freed before being put on freelist
            std::atomic_uint32_t metadata;
        };
    
        struct SharedState {
            void *memoryBase;
            void *jobMemory;
            Arena *arenas;
            std::atomic_uint32_t freeHead;
        };
    
        static constexpr size_t maxJobSize = 1024;
        static constexpr size_t maxJobAlignment = 128;
    
        Alloc(SharedState &shared);
        void * alloc(SharedState &shared, uint32_t num_bytes, uint32_t alignment);
        void dealloc(SharedState &shared, void *ptr, uint32_t num_bytes);
    
        // FIXME: fix InitAlloc ownership
        static SharedState makeSharedState(InitAlloc alloc,
                                           uint32_t num_arenas);
    
    private:
        static constexpr size_t arena_size_ = 4096;
    
        uint32_t cur_arena_;
        uint32_t next_arena_;
        uint32_t arena_offset_;
        uint32_t arena_used_bytes_;
    };

private:
    struct alignas(64) JobCounts {
        std::atomic_uint32_t numHigh;
        std::atomic_uint32_t numOutstanding;
    };

    JobManager(void *ctx_init_data,
               uint32_t num_ctx_init_bytes,
               uint32_t ctx_init_alignment,
               void (*ctx_init_fn)(void *, void *, WorkerInit &&),
               uint32_t num_ctx_bytes,
               uint32_t ctx_alignment,
               void (*start_fn)(Context &, void *),
               void *start_data,
               int desired_num_workers,
               int num_io,
               StateManager *state_mgr,
               bool pin_workers);

    struct Init;
    JobManager(const Init &init);

    inline void * allocJob(int worker_idx, uint32_t num_bytes,
                           uint32_t alignment);
    inline void deallocJob(int worker_idx, void *ptr, uint32_t num_bytes);

    JobID queueJob(int thread_idx, Job::EntryPtr job_func,
                   JobContainerBase *job_data, uint32_t num_invocations, 
                   uint32_t parent_job_idx,
                   JobPriority prio = JobPriority::Normal);

    JobID reserveProxyJobID(uint32_t parent_job_idx);
    void relinquishProxyJobID(uint32_t job_idx);

    void markJobFinished(int thread_idx, JobContainerBase *job,
                         uint32_t job_size);

    template <typename Fn>
    inline void addToRunQueue(int thread_idx, JobPriority prio, Fn &&add_cb);

    inline void addToWaitQueue(int thread_idx, Job::EntryPtr job_func,
        JobContainerBase *job_data, uint32_t num_invocations,
        JobPriority prio);

    inline void findReadyJobs(int thread_idx);

    inline bool getNextJob(void *queue_base, int thread_idx,
                           bool check_waiting, Job *job);

    inline void runJob(const int thread_idx, Context *ctx,
                       Job::EntryPtr fn, JobContainerBase *job_data,
                       uint32_t invocation_offset, uint32_t num_invocations);

    void workerThread(const int thread_idx, Context *ctx);
    void ioThread(const int thread_idx, Context *ctx);

    HeapArray<std::thread, InitAlloc> threads_;

    Alloc::SharedState alloc_state_;
    HeapArray<Alloc, InitAlloc> job_allocs_;

    void *const per_thread_data_;
    void *const high_base_;
    void *const normal_base_;
    void *const io_base_;
    void *const waiting_base_;
    void *const tracker_base_;
    std::counting_semaphore<> io_sema_;
    JobCounts job_counts_;
};

}

#include "job.inl"
