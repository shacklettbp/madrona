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
    uint32_t gen;
    uint32_t id;

    static constexpr inline JobID none();
};

struct JobContainerBase {
    JobID id;
#ifdef MADRONA_MW_MODE
    uint32_t worldID;
#endif
    uint32_t numDependencies;

    template <size_t N> struct DepsArray;
};

template <typename Fn, size_t N>
struct JobContainer : public JobContainerBase {
    [[no_unique_address]] DepsArray<N> dependencies;
    [[no_unique_address]] Fn fn;

#ifdef MADRONA_MW_MODE
    template <typename... DepTs>
    inline JobContainer(uint32_t world_id, Fn &&fn, DepTs ...deps);
#else
    template <typename... DepTs>
    inline JobContainer(Fn &&fn, DepTs ...deps);
#endif
};

struct Job {
    void *func;
    JobContainerBase *data;
    uint32_t invocationOffset;
    uint32_t numInvocations;
};

class JobManager {
public:
    template <typename StartFn> struct EntryConfig;

    template <typename ContextT, typename DataT, typename StartFn>
    static EntryConfig<StartFn> makeEntry(StartFn &&start_fn);

    template <typename StartFn>
    JobManager(const EntryConfig<StartFn> &entry_cfg,
               int desired_num_workers,
               int num_io,
               StateManager *state_mgr,
               bool pin_workers = true);

    ~JobManager();

    inline JobID reserveProxyJobID(int thread_idx, JobID parent_id);
    inline void relinquishProxyJobID(int thread_idx, JobID job_id);

    template <typename ContextT, bool single_invoke, typename Fn,
              typename... DepTs>
    JobID queueJob(int thread_idx,
                   Fn &&fn,
                   uint32_t num_invocations,
                   JobID parent_id,
                   MADRONA_MW_COND(uint32_t world_id,)
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

    struct RunQueue {
        std::atomic_uint32_t head;
        std::atomic_uint32_t correction;
        std::atomic_uint32_t auth;
        char pad[MADRONA_CACHE_LINE - sizeof(std::atomic_uint32_t) * 3];
        std::atomic_uint32_t tail;
    };

private:
    template <typename ContextT, typename ContainerT>
    static void singleInvokeEntry(Context *ctx_base, JobContainerBase *data);

    template <typename ContextT, typename ContainerT>
    static void multiInvokeEntry(Context *ctx_base, JobContainerBase *data,
                                 uint64_t invocation_offset,
                                 uint64_t num_invocations,
                                 RunQueue *thread_queue);

    using SingleInvokeFn = decltype(&singleInvokeEntry<Context, void>);
    using MultiInvokeFn = decltype(&multiInvokeEntry<Context, void>);

    JobManager(uint32_t num_ctx_userdata_bytes,
               uint32_t ctx_userdata_alignment,
               void (*ctx_init_fn)(void *, void *, WorkerInit &&),
               uint32_t num_ctx_bytes,
               uint32_t ctx_alignment,
               void (*start_fn)(Context *, void *),
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

    JobID queueJob(int thread_idx, void *job_func,
                   JobContainerBase *job_data, uint32_t num_invocations, 
                   uint32_t parent_job_idx,
                   JobPriority prio = JobPriority::Normal);

    JobID reserveProxyJobID(int thread_idx, uint32_t parent_job_idx);
    void relinquishProxyJobID(int thread_idx, uint32_t job_idx);

    inline void jobFinishedImpl(int thread_idx, uint32_t job_idx);
    void jobFinished(int thread_idx, uint32_t job_idx);

    bool markInvocationsFinished(int thread_idx,
                                 JobContainerBase *job,
                                 uint32_t num_invocations);

    template <typename Fn>
    inline void addToRunQueue(int thread_idx, JobPriority prio, Fn &&add_cb);

    inline void addToWaitQueue(int thread_idx, void *job_func,
        JobContainerBase *job_data, uint32_t num_invocations,
        JobPriority prio);

    inline void findReadyJobs(int thread_idx);

    inline bool isQueueEmpty(uint32_t head, uint32_t correction,
                             uint32_t tail) const;

    inline uint32_t dequeueJobIndex(RunQueue *run_queue);

    inline bool getNextJob(void *queue_base, int thread_idx,
                           bool check_waiting, Job *job);

    void splitJob(MultiInvokeFn fn_ptr, JobContainerBase *job_data,
                  uint32_t invocation_offset, uint32_t num_invocations,
                  RunQueue *run_queue);

    inline void runJob(const int thread_idx, Context *ctx_ptr,
                       void *fn, JobContainerBase *job_data,
                       uint32_t invocation_offset, uint32_t num_invocations);

    void workerThread(const int thread_idx,
                      void *context_base,
                      uint32_t num_context_bytes);

    void ioThread(const int thread_idx,
                  void *context_base,
                  uint32_t num_context_bytes);

    HeapArray<std::thread, InitAlloc> threads_;

    Alloc::SharedState alloc_state_;
    HeapArray<Alloc, InitAlloc> job_allocs_;

    void *const per_thread_data_;
    void *const high_base_;
    void *const normal_base_;
    void *const io_base_;
    void *const waiting_base_;
    void *const tracker_base_;
    void *const tracker_cache_base_;
    std::counting_semaphore<> io_sema_;
    alignas(MADRONA_CACHE_LINE) std::atomic_uint32_t num_high_;
    alignas(MADRONA_CACHE_LINE) std::atomic_uint32_t num_outstanding_;
};

}

#include "job.inl"
