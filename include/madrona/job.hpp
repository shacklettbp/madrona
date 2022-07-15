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

class JobID {
private:
    JobID(uint64_t id) : id_(id) {}
    uint64_t id_;

friend class JobManager;
};

class Job {
private:
    using EntryPtr = void (*)(Context &, void *);
    EntryPtr func_;
    void *data_;

friend class Context;
friend class JobManager;
};

class JobManager {
public:
    class Alloc;

    template <typename Fn>
    JobManager(int desired_num_workers, int num_io, StateManager &state_mgr,
               void *world_data, Fn &&fn, bool pin_workers = true);
    ~JobManager();

    inline void * allocJob(int worker_idx, uint32_t num_bytes,
                           uint32_t alignment);
    inline void deallocJob(int worker_idx, void *ptr, uint32_t num_bytes);

    inline JobID queueJob(int thread_idx, Job job,
                          const JobID *deps, uint32_t num_dependencies,
                          JobPriority prio = JobPriority::Normal);
    JobID queueJobs(int thread_idx, const Job *jobs, uint32_t num_jobs,
                    const JobID *deps, uint32_t num_dependencies,
                    JobPriority prio = JobPriority::Normal);

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

    JobManager(int desired_num_workers, int num_io, Job::EntryPtr start_func,
               void *start_data, StateManager &state_mgr, void *world_data,
               bool pin_workers);

    JobID queueJob(int thread_idx, Job::EntryPtr job_func, void *job_data,
                   const JobID *deps, uint32_t num_dependencies,
                   JobPriority prio);

    void workerThread(const int thread_idx, StateManager &state_mgr,
                      void *world_data);
    void ioThread(const int thread_idx, StateManager &state_mgr,
                  void *world_data);

    HeapArray<std::thread, InitAlloc> threads_;

    Alloc::SharedState alloc_state_;
    HeapArray<Alloc, InitAlloc> job_allocs_;

    void *const queue_store_;
    void *const high_start_;
    void *const normal_start_;
    void *const io_start_;
    void *const waiting_start_;
    std::counting_semaphore<> io_sema_;
    JobCounts job_counts_;
};

}

#include "job.inl"
