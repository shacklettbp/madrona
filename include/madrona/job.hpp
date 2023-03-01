/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <madrona/fwd.hpp>
#include <madrona/memory.hpp>
#include <madrona/heap_array.hpp>
#include <madrona/span.hpp>
#include <madrona/sync.hpp>

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
    int32_t id;

    static constexpr inline JobID none();
};

struct JobContainerBase {
    JobID id;
    uint32_t jobSize;
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

    template <typename... DepTs>
    inline JobContainer(uint32_t job_size, MADRONA_MW_COND(uint32_t world_id,)
                        Fn &&fn, DepTs ...deps);
};

struct Job {
    void (*func)();
    JobContainerBase *data;
    uint32_t invocationOffset;
    uint32_t numInvocations;
};

class JobManager {
public:
    template <typename StartFn, typename UpdateFn> struct EntryConfig;

    template <typename ContextT, typename StartFn>
    static EntryConfig<StartFn, void (*)(Context *, void *)> makeEntry(
        StartFn &&start_fn);

    template <typename ContextT, typename StartFn, typename UpdateFn>
    static EntryConfig<StartFn, UpdateFn> makeEntry(StartFn &&start_fn,
                                                    UpdateFn &&update_fn);

    template <typename StartFn, typename UpdateFn>
    JobManager(const EntryConfig<StartFn, UpdateFn> &entry_cfg,
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
            AtomicU32 metadata;
        };
    
        struct SharedState {
            void *memoryBase;
            void *jobMemory;
            Arena *arenas;
            AtomicU32 freeHead;
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
        AtomicU32 head;
        AtomicU32 correction;
        AtomicU32 auth;
        char pad[MADRONA_CACHE_LINE - sizeof(AtomicU32) * 3];
        AtomicU32 tail;
    };

private:
    struct SchedulerState {
        uint32_t numWaiting;
        uint32_t numSleepingWorkers;
        SpinLock lock;
    };

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
               void *start_fn_data,
               void (*update_fn)(Context *, void *),
               void *update_fn_data,
               int desired_num_workers,
               int num_io,
               StateManager *state_mgr,
               bool pin_workers);

    struct Init;
    JobManager(const Init &init);

    inline void * allocJob(int worker_idx, uint32_t num_bytes,
                           uint32_t alignment);
    inline void deallocJob(int worker_idx, void *ptr, uint32_t num_bytes);

    JobID queueJob(int thread_idx, void (*job_func)(),
                   JobContainerBase *job_data, uint32_t num_invocations, 
                   uint32_t parent_job_idx,
                   JobPriority prio = JobPriority::Normal);

    JobID reserveProxyJobID(int thread_idx, uint32_t parent_job_idx);
    void relinquishProxyJobID(int thread_idx, uint32_t job_idx);

    void markInvocationsFinished(int thread_idx,
                                 JobContainerBase *job_data,
                                 int32_t job_idx,
                                 uint32_t num_invocations);

    inline JobID getNewJobID(int thread_idx, uint32_t parent_job_idx, 
                             uint32_t num_invocations);

    template <typename Fn>
    inline void addToRunQueue(int thread_idx, JobPriority prio, Fn &&add_cb);

    inline void addToWaitQueue(int thread_idx, void (*job_func)(),
        JobContainerBase *job_data, uint32_t num_invocations,
        JobPriority prio);

    enum class WorkerControl : uint64_t;
    inline WorkerControl schedule(int thread_idx, Job *run_job);

    inline bool isQueueEmpty(uint32_t head, uint32_t correction,
                             uint32_t tail) const;

    inline uint32_t dequeueJobIndex(RunQueue *run_queue);

    inline WorkerControl getNextJob(void *queue_base, int thread_idx,
                                    int init_search_idx,
                                    bool run_scheduler,
                                    Job *job);

    inline WorkerControl tryScheduling(JobManager::WorkerControl default_ctrl,
                                       int thread_idx, Job *next_job);

    inline bool shouldSplitJob(RunQueue *queue) const;

    void splitJob(MultiInvokeFn fn_ptr, JobContainerBase *job_data,
                  uint32_t invocation_offset, uint32_t num_invocations,
                  RunQueue *run_queue);

    inline void runJob(const int thread_idx, Context *ctx,
                       void (*generic_fn)(), JobContainerBase *job_data,
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

    SchedulerState scheduler_;
    void *const state_ptr_;
    void *const high_base_;
    void *const normal_base_;
    void *const io_base_;
    void *const tracker_base_;
    void *const tracker_cache_base_;
    void *const worker_base_;
    void *const log_base_;
    void *const waiting_jobs_;
    uint32_t num_compute_workers_;

    std::counting_semaphore<> io_sema_;
    alignas(MADRONA_CACHE_LINE) AtomicU32 num_high_;
};

}

#include "job.inl"
