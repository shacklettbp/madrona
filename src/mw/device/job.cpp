/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include <atomic>
#include <array>
#include <cuda/barrier>

#include <madrona/context.hpp>
#include <madrona/utils.hpp>
#include <madrona/memory.hpp>

#include "mw_gpu/const.hpp"
#include "mw_gpu/cu_utils.hpp"

// Considerations:
//  - Enforcing end-user data safety constraints around iteration & creation / deletion: fundamentally, a job needs to not run until it can be run by all worlds.
//  - This means a given launch should have all the union of all dependencies of all instances of the same launch across worlds. Practically speaking, this means all the children launches of a given parent need to be deferred until the parent has completed across all worlds. Then a merging step has to run, where children jobs that match across worlds need to be grouped together into single launches with shared dependencies.
//  - Challenges:
//      - Currently job "uniqueness" isn't enforced. This means job A can launch the same function pointer twice, and the second launch can depend on the first. This could be changed, but may be jarring given the current lambda API
//      - The above issue implies that searching through the list of children, merging based on function ptr alone is not enough. One solution is (func_ptr, # of launches of ptr in this job). That's annoying because now you need to track per func ptr state while building 
//  - Advantages:
//      - This scheme enforces maximum merging of jobs. If this wasn't the case, there would need to be some other mechanism to recover common jobs in conditional execution cases.
//  - Disadvantages:
//      - Lots of synchronization, less ability to overlap. Partly lose a major advantage of the megakernel which is that there is now no driver mandated synchronization
//  - Sketch:
//      - Every running job has a num_worlds wide log array.
//

using std::atomic_uint32_t;
using std::memory_order;

extern "C" {
__global__ void madronaMWGPUMegakernel(uint32_t func_id, madrona::JobContainerBase *data, uint32_t *data_indices, uint32_t *invocation_offsets, uint32_t num_launches, uint32_t grid);
}

namespace madrona {
namespace mwGPU {

namespace consts {

static constexpr uint32_t allActive = 0xFFFFFFFF;
static constexpr uint32_t jobTrackerTerm = ~0u;

static constexpr uint32_t jobsPerWaitQueue = 16384;
static constexpr uint32_t numWaitQueues = 64;

}

struct JobTracker {
    uint32_t parent;
    std::atomic_uint32_t numOutstanding;
    std::atomic_uint32_t remainingInvocations;
};

struct ThreadblockData {
    ChunkAllocator::Cache chunkCache;
    std::atomic_uint32_t curChunkID;
    std::atomic_uint32_t remainingChunkBytes;
};

static __shared__ ThreadblockData tbData;

struct WaitQueue {
    SpinLock lock;
    uint32_t numWaiting;
    Job waitingJobs[consts::jobsPerWaitQueue];
};

static inline WaitQueue * getWaitQueues(JobManager *mgr, uint32_t idx)
{
    return (WaitQueue *)((char *)mgr + GPUImplConsts::get().waitQueueOffset);
}

static inline Job *getBaseJobList(JobManager *mgr)
{
    return (Job *)((char *)mgr + GPUImplConsts::get().jobListOffset);
}

static void initializeJobSystem(JobManager *mgr)
{
    if (threadIdx.x == 0) {
        mgr->numOutstandingInvocations = 0;

        for (int i = 0; i < 8; i++) {
            mgr->activeGrids[i] = 0;
        }

        mgr->freeTrackerHead.store(JobID { 0, 0 }, std::memory_order_relaxed);
    }

    __syncthreads();

    auto grids = getGridInfo(mgr);

    for (int i = threadIdx.x; i < (int)consts::numJobGrids; i += blockDim.x) {
        grids[i].waitJobHead.store(0, std::memory_order_relaxed);
        grids[i].waitJobTail.store(0, std::memory_order_relaxed);
        grids[i].waitQueueLock.store(0, std::memory_order_relaxed);
        grids[i].numRunning.store(0, std::memory_order_relaxed);
    }

    auto shared_trackers = getSharedJobTrackers(mgr);

    int num_shared_trackers =
        GPUImplConsts::get().maxJobsPerGrid * consts::numJobGrids;

    for (int i = threadIdx.x; i < num_trackers; i += blockDim.x) {
        JobTracker &tracker = trackers[i];
        tracker.gen.store(0, std::memory_order_relaxed);
        if (i < num_trackers - 1) {
            tracker.parent.store(i + 1, std::memory_order_relaxed);
        } else {
            tracker.parent.store(consts::jobTrackerTerm, std::memory_order_relaxed);
        }
        tracker.numOutstanding.store(0, std::memory_order_relaxed);
    }

    std::atomic_thread_fence(std::memory_order_release);
}

static inline const JobID * getJobDependencies(JobContainerBase *job_base)
{
    return (const JobID *)((char *)job_base + sizeof(JobContainerBase));
}

static inline bool isJobReady(JobManager *job_mgr, Job &job)
{
    const JobTracker *trackers = getJobTrackers(job_mgr);
    for (int job_idx = 0; job_idx < job.numCombinedJobs; job_idx++) {
        JobContainerBase *job_data = (JobContainerBase *)(
            (char *)job.data + job.numBytesPerJob * job_idx);

        int num_deps = job_data->numDependencies;

        const JobID *dependencies = getJobDependencies(job_data);
        for (int i = 0; i < num_deps; i++) {
            JobID dependency = dependencies[i];

            if (trackers[dependency.id].gen == dependency.gen) {
                return false;
            }
        }
    }

    return true;
}

static inline void *allocateJobData(uint32_t total_bytes)
{
    // FIXME
    return malloc(total_bytes);
}

static inline void freeJobData(void *data)
{
    free(data);
}

static inline bool checkGEWrapped(uint32_t a, uint32_t b)
{
    return uint32_t(a - b) <= (1u << 31u);
}

static inline bool checkLTWrapped(uint32_t a, uint32_t b)
{
    return uint32_t(a - b) > (1u << 31u);
}

static inline uint32_t wrapQueueIdx(uint32_t idx, uint32_t num_elems)
{
    // Assumes num_elems is a power of 2
    return idx & (num_elems - 1);
}

// Only a single thread block can run this function
static void jobLoop()
{
    uint32_t lane_id = threadIdx.x % consts::numWarpThreads;
    uint32_t warp_id = threadIdx.x / consts::numWarpThreads;

    constexpr uint32_t total_num_warps = consts::numJobSystemKernelThreads /
        consts::numWarpThreads;

    constexpr uint32_t grids_per_warp =
        consts::numJobGrids / total_num_warps;

    auto job_mgr = getJobManager();
    auto base_job_grids = getGridInfo(job_mgr);
    auto base_job_list = getBaseJobList(job_mgr);

    const uint32_t max_jobs_per_grid = GPUImplConsts::get().maxJobsPerGrid;

    static __shared__ uint32_t active_grids_tmp[8];

    cuda::barrier<cuda::thread_scope_thread> cpy_barrier;
    init(&cpy_barrier, 1);

    auto findFirstReadyJob = [job_mgr, lane_id, max_jobs_per_grid](
        Job *job_list, uint32_t job_head, uint32_t job_tail) {
        int first_job_idx = -1;
        if (lane_id == 0) {
            for (uint32_t job_idx = job_head;
                 checkLTWrapped(job_idx, job_tail); job_idx++) {
                int wrapped_idx =
                    (int)wrapQueueIdx(job_idx, max_jobs_per_grid);
                Job &cur_job = job_list[wrapped_idx];

                if (isJobReady(job_mgr, cur_job)) {
                    first_job_idx = wrapped_idx;
                    break;
                }
            }
        }

        return __shfl_sync(consts::allActive, first_job_idx, 0);
    };

    auto getFreeGrid = [lane_id]() {
        int run_grid_idx = -1;
        if (lane_id == 0) {
#pragma unroll
            for (int bitfield_idx = 0; bitfield_idx < 8; bitfield_idx++) {
                uint32_t *grid_bitfield_ptr = &active_grids_tmp[bitfield_idx];

                uint32_t old_bitfield, set_bitfield;
                do {
                    old_bitfield = *grid_bitfield_ptr;

                    uint32_t inverse = ~old_bitfield;

                    // All grids running
                    if (inverse == 0) {
                        run_grid_idx = -1;
                        break;
                    }

                    uint32_t idx = 31 - __clz(inverse);

                    uint32_t mask = 1 << idx;

                    set_bitfield = old_bitfield | mask;

                    run_grid_idx = idx;
                } while (atomicCAS(grid_bitfield_ptr, old_bitfield,
                                   set_bitfield));

                if (run_grid_idx != -1) {
                    run_grid_idx = bitfield_idx * 32 + run_grid_idx;

                    break;
                }
            }
        }

        return __shfl_sync(consts::allActive, run_grid_idx, 0);
    };

    auto nextLoopSetup = [job_mgr, warp_id, lane_id]() {
        std::atomic_thread_fence(std::memory_order_acquire);

        if (warp_id == 0 && lane_id == 0) {
            for (int i = 0; i < 8; i++) {
                active_grids_tmp[i] = job_mgr->activeGrids[i];
            }
        }

        __syncthreads();
    };

    nextLoopSetup();
    while (true) {
        for (int wait_grid_offset = 0; wait_grid_offset < (int)grids_per_warp;
             wait_grid_offset++) {
            uint32_t wait_grid_idx =
                warp_id * grids_per_warp + wait_grid_offset;

            JobGridInfo &wait_grid = base_job_grids[wait_grid_idx];
            Job *waiting_jobs = base_job_list +
                wait_grid_idx * max_jobs_per_grid;

            // Relaxed is safe for head & tail, because
            // nextLoopSetup() does an acquire barrier
            uint32_t job_head =
                wait_grid.waitJobHead.load(std::memory_order_relaxed);
            // Cache the value of job tail, and use it across the warp,
            // it can be incremented by other threads
            uint32_t job_tail;
            if (lane_id == 0) {
                job_tail =
                    wait_grid.waitJobTail.load(std::memory_order_relaxed);
            }
            job_tail = __shfl_sync(consts::allActive, job_tail, 0);

            int first_job_idx =
                findFirstReadyJob(waiting_jobs, job_head, job_tail);
            if (first_job_idx == -1) {
                continue;
            }

            int run_grid_idx = getFreeGrid();
            if (run_grid_idx == -1) {
                break;
            }

            JobGridInfo &run_grid = base_job_grids[run_grid_idx];

            uint32_t first_func_id = waiting_jobs[first_job_idx].funcID;
            auto isJobMergable = [first_func_id, job_mgr](Job &job) {
                return job.funcID == first_func_id && isJobReady(job_mgr, job);
            };

            uint32_t num_bytes_per_job =
                waiting_jobs[first_job_idx].numBytesPerJob;

            uint32_t total_num_jobs = 0;
            uint32_t total_num_invocations = 0;

            // Could start from the unwrapped version of first_job_idx,
            // but would need to change findFirstReadyJob to return
            // a separate failure boolean
            for (uint32_t job_offset = job_head;
                 checkLTWrapped(job_offset, job_tail);
                 job_offset += consts::numWarpThreads) {
                uint32_t job_idx = job_offset + lane_id;

                bool inbounds = checkLTWrapped(job_idx, job_tail);

                // Force out of bounds indices in bounds
                if (!inbounds) {
                    job_idx = job_offset;
                }

                Job cur_job =
                    waiting_jobs[wrapQueueIdx(job_idx, max_jobs_per_grid)];

                bool merge_job = inbounds && isJobMergable(cur_job);
                uint32_t merge_mask =
                    __ballot_sync(consts::allActive, merge_job);

                uint32_t cur_num_jobs =
                    merge_job ? cur_job.numCombinedJobs : 0_u32;

                uint32_t num_prior_jobs = total_num_jobs +
                    warpExclusiveScan(lane_id, cur_num_jobs);

                // Copy job data into grid's run buffer
                if (merge_job) {
                    cuda::memcpy_async(run_grid.runData.buf + 
                        num_prior_jobs * num_bytes_per_job, cur_job.data,
                        num_bytes_per_job * cur_num_jobs, cpy_barrier);

                    assert((num_prior_jobs + cur_num_jobs) * num_bytes_per_job <=
                           1024 * 1024);
                } 

                // FIXME: this is a potentially massive loop that one thread
                // has to deal with
                uint32_t combined_num_invocations = 0;
                for (int job_idx = 0; job_idx != (int)cur_num_jobs; job_idx++) {
                    JobContainerBase *job_data = (JobContainerBase *)(
                        (char *)cur_job.data + job_idx * num_bytes_per_job);

                    uint32_t num_invocations = job_data->numInvocations;
                    combined_num_invocations += num_invocations;
                }

                uint32_t num_prior_invocations = total_num_invocations +
                    warpExclusiveScan(lane_id, combined_num_invocations);

                uint32_t num_setup_invocations = 0;
                for (int job_idx = 0; job_idx != (int)cur_num_jobs; job_idx++) {
                    JobContainerBase *job_data = (JobContainerBase *)(
                        (char *)cur_job.data + job_idx * num_bytes_per_job);

                    uint32_t num_invocations = job_data->numInvocations;

                    for (int i = 0; i != (int)num_invocations; i++) {
                        run_grid.jobDataIndices[(int)num_prior_invocations +
                            num_setup_invocations] = num_prior_jobs + job_idx;
                        run_grid.jobInvocationOffsets[(int)num_prior_invocations +
                            num_setup_invocations] = i;

                        num_setup_invocations++;

                        assert(num_prior_invocations + num_setup_invocations <
                               65536 * 16);
                    }
                }

                total_num_invocations =
                    num_prior_invocations + combined_num_invocations;
                total_num_jobs = num_prior_jobs + cur_num_jobs;

                // Get current running total of jobs merged together for launch
                uint32_t top_merge_thread = getHighestSetBit(merge_mask);
                total_num_invocations = __shfl_sync(consts::allActive,
                    total_num_invocations, top_merge_thread);
                total_num_jobs = __shfl_sync(consts::allActive,
                    total_num_jobs, top_merge_thread);
            }

            // Wait for all the async copies
            cpy_barrier.arrive_and_wait();

            __syncwarp();

            uint32_t base_wait_coalesce_idx = job_tail - 1u;

            // Free job data for jobs that have been copied into the run
            // data block, coalesce waiting job list
            // Unfortunately, this loop is in reverse, because the list needs
            // to be coalesced into the tail
            for (uint32_t job_offset = job_tail - 1u;
                 checkGEWrapped(job_offset, job_head);
                 job_offset -= consts::numWarpThreads) {

                uint32_t job_idx = job_offset - lane_id;

                bool inbounds = checkGEWrapped(job_idx, job_head);

                if (!inbounds) {
                    job_idx = job_offset;
                }

                Job cur_job =
                    waiting_jobs[wrapQueueIdx(job_idx, max_jobs_per_grid)];

                bool mergable = isJobMergable(cur_job);
                bool coalesceable = inbounds && !mergable;

                if (inbounds && mergable) {
                    freeJobData(cur_job.data);
                } 

                // The sync here also ensures all threads are done
                // using cur_job before any pointers are overwritten
                // when coalescing the list
                uint32_t coalesce_mask =
                    __ballot_sync(consts::allActive, coalesceable);

                // Coalesce jobs that won't be launched to make the waiting
                // list contiguous
                if (coalesceable) {
                    // Lower threads in the warp are farther ahead in the
                    // array due to reading backwards
                    uint32_t coalesce_idx = base_wait_coalesce_idx -
                        getNumLowerSetBits(coalesce_mask, lane_id);

                    if (coalesce_idx != job_idx) {
                        int wrapped_coalesce_idx =
                            wrapQueueIdx(coalesce_idx, max_jobs_per_grid);
                        waiting_jobs[wrapped_coalesce_idx] = cur_job;
                    }

                    base_wait_coalesce_idx = coalesce_idx - 1u;
                }

                if (coalesce_mask != 0) {
                    uint32_t top_coalesce_thread =
                        getHighestSetBit(coalesce_mask);
                    base_wait_coalesce_idx = __shfl_sync(consts::allActive,
                        base_wait_coalesce_idx, top_coalesce_thread);
                }
            }


            if (lane_id == 0) {
                wait_grid.waitJobHead.store(base_wait_coalesce_idx + 1,
                                            std::memory_order_relaxed);
            }

            std::atomic_thread_fence(std::memory_order_release);

            if (lane_id == 0) {
                uint32_t num_blocks = utils::divideRoundUp(total_num_invocations,
                    consts::numJobLaunchKernelThreads);
                
                run_grid.numRunning.store(total_num_invocations,
                                          std::memory_order_relaxed);

                madronaMWGPUMegakernel<<<num_blocks, consts::numJobLaunchKernelThreads>>>(
                    first_func_id,
                    (JobContainerBase *)run_grid.runData.buf,
                    run_grid.jobDataIndices, run_grid.jobInvocationOffsets,
                    total_num_invocations, run_grid_idx);
            }
        }

        __nanosleep(500);

        // Call this function at the end of the loop, in order to use
        // the same __syncthreads / acquire call for reading the activeGrid
        // bitfields and checking numOutstandingInvocations
        nextLoopSetup();

        if (job_mgr->numOutstandingInvocations == 0) {
            break;
        }
    }
}

static inline uint32_t computeMaxNumJobs(uint32_t num_worlds)
{
    // FIXME: scaling linearly like this probably doesn't make sense
    return consts::maxNumJobsPerWorld * num_worlds;
}

static inline JobID allocateJobTrackerSlot(JobManager *job_mgr,
                                           JobTracker *trackers)
{
    JobID cur_head = 
        job_mgr->freeTrackerHead.load(std::memory_order_acquire);

    JobID new_head;

    do {
        if (cur_head.id == consts::jobTrackerTerm) {
            break;
        }

        new_head.gen = cur_head.gen + 1;
        new_head.id = trackers[cur_head.id].parents[0];
    } while (!job_mgr->freeTrackerHead.compare_exchange_weak(
        cur_head, new_head, memory_order::release,
        memory_order::acquire));

    uint32_t job_id = cur_head.id;

    // FIXME
    if (job_id == consts::jobTrackerTerm) {
        assert(false);
    }

    uint32_t gen = tracker.gen.load(std::memory_order_relaxed);

    return JobID {
        gen,
        job_id,
    };
}

static inline void freeJobTrackerSlot(uint32_t job_id)
{
    auto job_mgr = getJobManager();
    JobTracker *trackers = getJobTrackers(job_mgr);

    JobTracker &tracker = trackers[job_id];
    tracker.gen = tracker.gen + 1;

    JobID new_head;
    new_head.id = job_id;

    JobID cur_head = job_mgr->freeTrackerHead.load(
        std::memory_order_relaxed);

    do {
        new_head.gen = cur_head.gen + 1;

        tracker.parents[0] = cur_head.id;
    } while (!job_mgr->freeTrackerHead.compare_exchange_weak(
           cur_head, new_head,
           memory_order::release, memory_order::relaxed));
}

static inline void decrementJobTracker(JobTracker *job_trackers, JobID job_id)
{
    uint32_t cur_id = job_id.id;
    while (cur_id != consts::jobTrackerTerm) {
        JobTracker &tracker = job_trackers[cur_id];

        uint32_t prev_outstanding =
            tracker.numOutstanding.fetch_sub(1, std::memory_order_acq_rel);

        if (prev_outstanding == 1 &&
            tracker.remainingInvocations.load(std::memory_order_relaxed) == 0) {
            uint32_t parent = tracker.parent;

            freeJobTrackerSlot(cur_id);

            cur_id = parent;
        } else {
            break;
        }
    }
}

// This function should only be called by the wave leader
static inline void queueMultiJobInWaitList(
    uint32_t func_id, JobContainerBase *data, uint32_t grid_id,
    uint32_t num_jobs, uint32_t total_num_invocations,
    uint32_t num_bytes_per_job)
{
    Job job {
        .data = data,
        .funcID = func_id,
        .numCombinedJobs = num_jobs,
        .numBytesPerJob = num_bytes_per_job,
    };

    auto job_mgr = getJobManager();

    const auto base_job_grids = getGridInfo(job_mgr);
    const auto base_job_list = getBaseJobList(job_mgr);
    const uint32_t max_jobs_per_grid = GPUImplConsts::get().maxJobsPerGrid;

    JobGridInfo &cur_grid = base_job_grids[grid_id];
    Job *job_list = base_job_list + grid_id * max_jobs_per_grid;

    // Get lock
    while (cur_grid.waitQueueLock.exchange(1, std::memory_order_acq_rel)) {}

    uint32_t cur_job_pos = cur_grid.waitJobTail.load(std::memory_order_acquire);
    job_list[wrapQueueIdx(cur_job_pos, max_jobs_per_grid)] = job;

    cur_grid.waitJobTail.fetch_add(1, std::memory_order_relaxed);

    cur_grid.waitQueueLock.store(0, std::memory_order_relaxed);
    
    atomicAdd(&job_mgr->numOutstandingInvocations, total_num_invocations);

    std::atomic_thread_fence(std::memory_order_release);
}

}

bool JobManager::startBlockIter(uint32_t block_idx, RunnableJob *out_job)
{
}

void JobManager::finishBlockIter(uint32_t block_idx)
{
    __syncthreads(); // Ensure entire threadblock has finished
    uint32_t num_log_entries = 
        tbRunData.numLogEntries.load(memory_order::relaxed);
   
    for (int i = threadIdx.x; i < (int)num_log_entries;
         i += mwGPU::consts::numMegakernelThreads) {
        uint8_t offset_lp = tbRunData.dataTmpOffsets[i];
        int offset = (int)offset_lp * 8;

        LogEntry *log = (LogEntry *)(tbRunData.dataTmp + offset);

    }

    __syncthreads();


    if (threadIdx.x == 0) {
        tbRunData.numLogEntries.store(0, memory_order::relaxed);
    }
}

Context::WaveInfo Context::computeWaveInfo()
{
    using namespace mwGPU;

    uint32_t active = __activemask();
    uint32_t num_active = __popc(active);

    uint32_t coalesced_idx = getNumLowerSetBits(active, lane_id_);

    return WaveInfo {
        .activeMask = active,
        .numActive = num_active,
        .coalescedIDX = coalesced_idx,
    };
}

void Context::stageChildJob(uint32_t func_id, uint32_t num_combined_jobs,
                            uint32_t bytes_per_job, void *containers)
{
    uint32_t offset = 
        tbScratch.waitQueue.numWaiting.fetch_add(1, memory_order::relaxed);

    tbScratch.waitingJobs[offset] = Job {
        .data = (JobContainerBase *)containers_tmp,
        .funcID = func_id,
        .numCombinedJobs = num_combined_jobs,
        .bytesPerJob = bytes_per_job,
    };
}

JobID Context::waveSetupNewJob(uint32_t func_id, bool link_parent,
        uint32_t num_invocations, uint32_t bytes_per_job,
        void **thread_data_store)
{
    auto wave_info = computeWaveInfo();
    auto job_mgr = JobManager::get();
    auto job_trackers = getJobTrackers(job_mgr);

    JobID child_id;
    char *tmp_data;

    uint32_t num_total_invocations =
        __reduce_add_sync(wave_info.activeMask, num_invocations);

    if (wave_info.isLeader()) {
        child_id = allocateJobTrackerSlot(job_mgr, job_trackers);
        job_trackers[child_id.id].numOutstanding.store(1, memory_order::relaxed);
        job_trackers[child_id.id].remainingInvocations.store(
            num_total_invocations, memory_order::relaxed);

        uint32_t total_bytes = wave_info.numActive * bytes_per_job;
        tmp_data = (char *)malloc(total_bytes);

        stageChildJob(func_id, tmp_data, bytes_per_job, wave_info.numActive);
    }

    child_id = {
        .gen = __shfl_sync(wave_info.activeMask, child_id.gen,
                           wave_info.leaderLane);
        .id = __shfl_sync(wave_info.activeMask, child_id.id,
                          wave_info.leaderLane);
    };

    uint32_t parent_id;
    if (link_parent) {
        parent_id = job_id_.id;

        uint32_t parent_match =
            __match_any_sync(wave_info.activeMask, parent_id);

        uint32_t top_thread = getHighestSetBit(parent_match);
        uint32_t num_children = __popc(parent_match);

        if (lane_id_ == top_thread) {
            trackers[parent_id].numOutstanding.fetch_add(num_children,
               std::memory_order_relaxed);
        }

        job_trackers[child_id.id].parents[wave_info.coalescedIDX] = parent_id;
    } else {
        parent_id = consts::jobTrackerTerm;
    }

    tmp_data = (char *)__shfl_sync(wave_info.activeMask, (uintptr_t)tmp_data,
                                   wave_info.leaderLane);

    *thread_data_store = tmp_data + wave_info.coalescedIDX * bytes_per_job;

    return child_id;
}

JobID Context::getNewJobID(bool link_parent, uint32_t num_invocations)
{
    using namespace mwGPU;

    auto job_mgr = getJobManager();
    JobTracker *trackers = getJobTrackers(job_mgr);

    uint32_t parent_id;
    if (link_parent) {
        parent_id = job_id_.id;
        trackers[parent_id].numOutstanding.fetch_add(1,
           std::memory_order_release);
    } else {
        parent_id = consts::jobTrackerTerm;
    }

    return allocateJobTrackerSlot(job_mgr, trackers, parent_id, num_invocations);
}

// Allocates a shared block of memory for the active threads in wave,
// where lower threads are given a pointer to an early chunk of the block
JobContainerBase * Context::allocJob(uint32_t bytes_per_job,
                                     WaveInfo wave_info)
{
    using namespace mwGPU;

    void *base_store;
    if (lane_id_ == wave_info.leaderLane) {
        base_store = allocateJobData(bytes_per_job * wave_info.numActive);
    }

    // Sync store point & id with wave
    base_store = (void *)__shfl_sync(wave_info.activeMask,
        (uintptr_t)base_store, wave_info.leaderLane);

    return (JobContainerBase *)(
        (char *)base_store + bytes_per_job * wave_info.coalescedIDX);
}

void Context::logNewJob(uint32_t func_id, JobContainerBase *data,
                        uint32_t num_invocations,
                        uint32_t num_bytes_per_job,
                        uint32_t lane_id,
                        WaveInfo wave_info)
{
    using namespace mwGPU;

    uint32_t total_num_invocations =
        warpSum(wave_info.activeMask, lane_id, num_invocations);

    if (lane_id_ == wave_info.leaderLane) {
        queueMultiJobInWaitList(func_id, data, grid_id_,
            wave_info.numActive, total_num_invocations,
            num_bytes_per_job);
    }
}

void Context::markJobFinished()
{
    using namespace mwGPU;
    auto job_mgr = JobManager::get();

    JobTracker *trackers = getJobTracker(job_mgr);

#pragma loop unroll
    uint32_t job_idx = job_id_.id;
    for (int i = 0; i < consts::numWarpThreads; i++) {
        uint32_t other_job_idx = __shfl_sync(consts::allActive, job_idx);

        uint32_t job_match = __match_any_sync(wave_info.activeMask, job_idx);

        uint32_t top_thread = getHighestSetBit(job_match);
        uint32_t num_invocations = __popc(job_match);

        if (lane_id_ == top_thread) {
            uint32_t prev_invocations = 
                trackers[job_idx].remainingInvocations.fetch_sub(num_invocations,
                    memory_order::relaxed);
        }

            if (prev_invocations == num_invocations) {
                decrementJobTracker(trackers, job_id_);
            }
        }
    }

    __syncthreads();

    uint32_t num_waiting =
        tbScratch.waitQueue.numWaiting.load(memory_order::relaxed);

    if (threadIdx.x == 0) {
        uint32_t prev_invocations = metadata.remainingInvocations.fetch_sub(
            num_invocations);

        if (prev_invocations == num_invocations) {
            scratch =
                metadata.numStaged.load(memory_order::relaxed);
        } else {
            scratch[0] = ~0_u32;
        }
    }

    __syncthreads();

    uint32_t num_staged = scratch[0];
    if (num_staged == ~0_u32) {
        return;
    }

    // Merge children jobs
    // FIXME: optimization - do a reduction at the thread block level first
    // Or, launch a job with num_staged invocations that does this reduction
    // globally

    StagedJob *staged_jobs = getStagedJobs(metadata);

    cuda::barrier<cuda::thread_scope_block> cpy_barrier;
    cuda::barrier::init(&cpy_barrier, consts::numMegakernelThreads);

    for (int i = 0; i < num_staged; i += consts::numMegakernelThreads) {
        int offset = i + threadIdx.x;
        bool inbounds = offset < num_staged;
        offset = inbounds ? offset : num_staged - 1;

        StagedJob &staged_job = staged_jobs[offset];
        uint32_t func_id = staged_job.funcID;

        __syncthreads();
    }
    
    cpy_barrier.arrive_and_wait();

    if (threadIdx.x == 0) {
        decrementJobTracker(job_mgr, merged_job_id_);
    }
}

static inline uint32_t computeNumAllocatorChunks(uint64_t num_bytes)
{
    return utils::divideRoundUp(num_bytes,
        (uint64_t)ChunkAllocator::chunkSize);
}

}

extern "C" __global__ void madronaMWGPUComputeConstants(
    uint32_t num_worlds,
    uint32_t num_world_data_bytes,
    uint32_t world_data_alignment,
    uint64_t num_allocator_bytes,
    madrona::mwGPU::GPUImplConsts *out_constants,
    size_t *job_system_buffer_size)
{
    using namespace madrona;
    using namespace madrona::mwGPU;

    uint32_t max_num_jobs_per_grid =
        madrona::mwGPU::computeMaxNumJobs(num_worlds);

    uint32_t max_num_jobs = consts::numJobGrids * max_num_jobs_per_grid;

    uint64_t total_bytes = sizeof(JobManager);

    uint64_t state_mgr_offset = utils::roundUp(total_bytes,
        (uint64_t)alignof(StateManager));

    total_bytes = state_mgr_offset + sizeof(StateManager);

    uint64_t chunk_allocator_offset = utils::roundUp(total_bytes,
        (uint64_t)alignof(ChunkAllocator));

    total_bytes = chunk_allocator_offset + sizeof(ChunkAllocator);

    uint64_t chunk_base_offset = utils::roundUp(total_bytes,
        (uint64_t)alignof(ChunkAllocator::chunkSize));

    uint64_t num_chunks = computeNumAllocatorChunks(num_allocator_bytes);
    total_bytes += num_chunks * ChunkAllocator::chunkSize;

    uint64_t world_data_offset =
        utils::roundUp(total_bytes, (uint64_t)world_data_alignment);

    total_bytes =
        world_data_offset + (uint64_t)num_world_data_bytes * num_worlds;

    uint64_t grid_offset = utils::roundUp(total_bytes,
        (uint64_t)alignof(JobGridInfo));

    total_bytes = grid_offset + sizeof(JobGridInfo) * consts::numJobGrids;

    uint64_t wait_job_offset = madrona::utils::roundUp(total_bytes,
        (uint64_t)alignof(Job));

    uint64_t num_job_bytes = sizeof(Job) * max_num_jobs;

    total_bytes = wait_job_offset + num_job_bytes;

    uint64_t tracker_offset = madrona::utils::roundUp(total_bytes,
        (uint64_t)alignof(JobTracker));

    // FIXME: using max_num_jobs for this doesn't quite make sense, because
    // there will be more outstanding trackers than waiting jobs due to
    // parent trackers remaining alive until children finish
    total_bytes = tracker_offset + sizeof(JobTracker) * max_num_jobs * 2;

    *out_constants = GPUImplConsts {
        .jobSystemAddr = (void *)0ul,
        .stateManagerAddr = (void *)state_mgr_offset,
        .chunkAllocatorAddr = (void *)chunk_allocator_offset,
        .chunkBaseAddr = (void *)chunk_base_offset,
        .worldDataAddr = (void *)world_data_offset,
        .numWorldDataBytes = num_world_data_bytes,
        .numWorlds = num_worlds,
        .jobGridsOffset = (uint32_t)grid_offset,
        .jobListOffset = (uint32_t)wait_job_offset,
        .maxJobsPerGrid = max_num_jobs_per_grid,
        .jobTrackerOffset = (uint32_t)tracker_offset,
    };

    *job_system_buffer_size = total_bytes;
}

extern "C" __global__  void madronaMWGPUInitialize(
    uint64_t num_allocator_bytes)
{
    using namespace madrona;
    using namespace madrona::mwGPU;

    auto job_mgr = getJobManager();

    if (threadIdx.x == 0) {
        new (job_mgr) JobManager();
    }
    __syncthreads();

    initializeJobSystem(job_mgr);

    if (threadIdx.x == 0) {
        new (GPUImplConsts::get().stateManagerAddr) StateManager(1024);
    }

    if (threadIdx.x == 0) {
        new (GPUImplConsts::get().chunkAllocatorAddr) ChunkAllocator(
            computeNumAllocatorChunks(num_allocator_bytes));
    }
}

extern "C" __global__ void madronaTrainJobSystemKernel()
{
    madrona::mwGPU::jobLoop();
}
