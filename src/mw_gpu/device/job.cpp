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

#include <madrona/mw_gpu/const.hpp>

extern "C" {
__constant__ madrona::mwGPU::GPUImplConsts madronaTrainGPUImplConsts;
}

namespace madrona {
namespace mwGPU {

namespace consts {

static constexpr uint32_t maxNumJobsPerWorld = 1024;
static constexpr uint32_t jobDataPerWorld = 65536;
static constexpr uint32_t numJobGrids = 256;
static constexpr uint32_t numJobSystemKernelThreads = 512;
static constexpr uint32_t allActive = ~0u;
static constexpr uint32_t jobTrackerTerm = ~0u;

}

struct alignas(64) RunData {
    char buf[1024 * 1024];
};

struct JobGridInfo {
    std::atomic_uint32_t waitJobHead;
    std::atomic_uint32_t waitJobTail;
    std::atomic_uint32_t waitQueueLock;
    std::atomic_uint32_t numRunning;
    RunData runData;
    uint32_t jobDataIndices[65536 * 16];
    uint32_t jobInvocationOffsets[65536 * 16];
};

struct alignas(16) JobTracker {
    std::atomic_uint32_t gen;
    std::atomic_uint32_t parent;
    std::atomic_uint32_t remainingInvocations;
    std::atomic_uint32_t numOutstanding;
};

static inline JobManager * getJobManager()
{
    return (JobManager *)GPUImplConsts::get().jobSystemAddr;
}

static inline JobGridInfo *getGridInfo(JobManager *mgr)
{
    return (JobGridInfo *)((char *)mgr + GPUImplConsts::get().jobGridsOffset);
}

static inline Job *getBaseJobList(JobManager *mgr)
{
    return (Job *)((char *)mgr + GPUImplConsts::get().jobListOffset);
}

static inline JobTracker *getJobTrackers(JobManager *mgr)
{
    return (JobTracker *)((char *)mgr + GPUImplConsts::get().jobTrackerOffset);
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

    auto trackers = getJobTrackers(mgr);

    int num_trackers = GPUImplConsts::get().maxJobsPerGrid * consts::numJobGrids;

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

static uint32_t warpSum(uint32_t mask, uint32_t lane_id, uint32_t val)
{
#pragma unroll
    for (int i = 16; i > 0; i /= 2) {
        uint32_t read_lane = lane_id ^ i;

        bool other_active = mask & (1 << read_lane);

        if (!other_active) {
            read_lane = lane_id;
        }

        uint32_t other = __shfl_sync(mask, val, read_lane);

        if (other_active) {
            val += other;
        }
    }

    return val;
}

static uint32_t warpInclusiveScan(uint32_t lane_id, uint32_t val)
{
#pragma unroll
    for (int i = 1; i < consts::numWarpThreads; i *= 2) {
        int tmp = __shfl_up_sync(consts::allActive, val, i);
        if ((int)lane_id >= i) {
            val += tmp;
        }
    }

    return val;
}

static inline uint32_t warpExclusiveScan(uint32_t lane_id, uint32_t val)
{
    return warpInclusiveScan(lane_id, val) - val;
}

static inline uint32_t getHighestSetBit(uint32_t mask)
{
    return mask == 0u ? 0u : 31u - __clz(mask);
}

static inline uint32_t getLowestSetBit(uint32_t mask)
{
    return mask == 0u ? 0u : __clz(__brev(mask));
}

static inline uint32_t getNumHigherSetBits(uint32_t mask, uint32_t idx)
{
    mask >>= idx + 1;

    return __popc(mask);
}

static inline uint32_t getNumLowerSetBits(uint32_t mask, uint32_t idx)
{
    mask <<= (32 - idx);

    return __popc(mask);
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

            Job::EntryPtr first_ptr = waiting_jobs[first_job_idx].fn;
            auto isJobMergable = [first_ptr, job_mgr](Job &job) {
                return job.fn == first_ptr && isJobReady(job_mgr, job);
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

                first_ptr<<<num_blocks, consts::numJobLaunchKernelThreads>>>(
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
                                           JobTracker *trackers,
                                           uint32_t parent_id,
                                           uint32_t num_invocations)
{
    JobID cur_head = 
        job_mgr->freeTrackerHead.load(std::memory_order_acquire);

    JobID new_head;

    do {
        if (cur_head.id == consts::jobTrackerTerm) {
            break;
        }

        new_head.gen = cur_head.gen + 1;
        new_head.id =
            trackers[cur_head.id].parent.load(std::memory_order_relaxed);
    } while (!job_mgr->freeTrackerHead.compare_exchange_weak(
        cur_head, new_head, std::memory_order_release,
        std::memory_order_acquire));

    uint32_t job_id = cur_head.id;

    // FIXME
    if (job_id == consts::jobTrackerTerm) {
        assert(false);
    }

    JobTracker &tracker = trackers[job_id];
    tracker.parent.store(parent_id, std::memory_order_relaxed);
    tracker.remainingInvocations.store(num_invocations,
                                       std::memory_order_release);
    tracker.numOutstanding.store(1, std::memory_order_release);
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
    tracker.gen.fetch_add(1, std::memory_order_acq_rel);

    JobID new_head;
    new_head.id = job_id;

    JobID cur_head = job_mgr->freeTrackerHead.load(
        std::memory_order_relaxed);

    do {
        new_head.gen = cur_head.gen + 1;

        tracker.parent.store(cur_head.id, std::memory_order_relaxed);
    } while (!job_mgr->freeTrackerHead.compare_exchange_weak(
           cur_head, new_head,
           std::memory_order_release, std::memory_order_relaxed));
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
            uint32_t parent = tracker.parent.load(std::memory_order_relaxed);

            freeJobTrackerSlot(cur_id);

            cur_id = parent;
        } else {
            break;
        }
    }
}

// This function should only be called by the wave leader
static inline void queueMultiJobInWaitList(
    Job::EntryPtr func, JobContainerBase *data, uint32_t grid_id,
    uint32_t num_jobs, uint32_t total_num_invocations,
    uint32_t num_bytes_per_job)
{
    Job job {
        .fn = func,
        .data = data,
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

Context::WaveInfo Context::computeWaveInfo()
{
    using namespace mwGPU;

    uint32_t active = __activemask();
    uint32_t num_active = __popc(active);

    uint32_t sel_idx = getLowestSetBit(active);
    uint32_t coalesced_idx = getNumLowerSetBits(active, lane_id_);

    return WaveInfo {
        .activeMask = active,
        .numActive = num_active,
        .leaderLane = sel_idx,
        .coalescedIDX = coalesced_idx,
    };
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

void Context::addToWaitList(Job::EntryPtr func, JobContainerBase *data,
                            uint32_t num_invocations,
                            uint32_t num_bytes_per_job,
                            uint32_t lane_id,
                            WaveInfo wave_info)
{
    using namespace mwGPU;

    uint32_t total_num_invocations =
        warpSum(wave_info.activeMask, lane_id, num_invocations);

    if (lane_id_ == wave_info.leaderLane) {
        queueMultiJobInWaitList(func, data, grid_id_,
            wave_info.numActive, total_num_invocations,
            num_bytes_per_job);
    }
}

void Context::markJobFinished(uint32_t num_invocations)
{
    using namespace mwGPU;

    auto job_mgr = getJobManager();
    JobTracker *job_trackers = getJobTrackers(job_mgr);

    // Num invocations above is the total invocations in the block.
    // Since the block may not all be using the same job id, need to
    // decrement remainingInvocations one at a time currently
    uint32_t prev_invocations =
        job_trackers[job_id_.id].remainingInvocations.fetch_sub(
            1, std::memory_order_acq_rel);

    if (prev_invocations == 1) {
        decrementJobTracker(job_trackers, job_id_);
    }

    __syncthreads();

    if (threadIdx.x != 0) return;

    JobGridInfo &cur_grid = getGridInfo(job_mgr)[grid_id_];

    atomicSub(&job_mgr->numOutstandingInvocations, num_invocations);

    uint32_t prev_running = cur_grid.numRunning.fetch_sub(num_invocations,
        std::memory_order_acq_rel);

    if (prev_running == num_invocations) {
        uint32_t grid_bitfield = grid_id_ / 32;
        uint32_t grid_bit = grid_id_ % 32;
        uint32_t mask = ~(1u << grid_bit);

        atomicAnd(&job_mgr->activeGrids[grid_bitfield], mask);
    }

    std::atomic_thread_fence(std::memory_order_release);
}

}

extern "C" __global__ void madronaTrainComputeGPUImplConstantsKernel(
    uint32_t num_worlds,
    uint32_t num_world_data_bytes,
    uint32_t world_data_alignment,
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

    uint64_t world_data_offset =
        utils::roundUp(total_bytes, (uint64_t)world_data_alignment);

    total_bytes =
        world_data_offset + (uint64_t)num_world_data_bytes * num_worlds;

    uint64_t grid_offset = utils::roundUp(total_bytes,
        (uint64_t)alignof(JobGridInfo));

    total_bytes = grid_offset + sizeof(JobGridInfo) * consts::numJobGrids;
    printf("%lu\n", sizeof(JobGridInfo));

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
        .worldDataAddr = (void *)world_data_offset,
        .numWorldDataBytes = num_world_data_bytes,
        .jobGridsOffset = (uint32_t)grid_offset,
        .jobListOffset = (uint32_t)wait_job_offset,
        .maxJobsPerGrid = max_num_jobs_per_grid,
        .jobTrackerOffset = (uint32_t)tracker_offset,
    };

    *job_system_buffer_size = total_bytes;
}

extern "C" __global__  void madronaTrainInitializeKernel()
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
}

extern "C" __global__ void madronaTrainJobSystemKernel()
{
    madrona::mwGPU::jobLoop();
}
