#include <atomic>
#include <array>
#include <cuda/barrier>

namespace std {

using namespace cuda::std;

}

#include <madrona/job.hpp>
#include <madrona/utils.hpp>

extern "C" {
__constant__ madrona::gpuTrain::JobSystemConstants madronaTrainJobSysConstants;
}

namespace madrona {
namespace gpuTrain {

namespace ICfg {

static constexpr uint32_t maxNumJobsPerWorld = 1024;
static constexpr uint32_t jobDataPerWorld = 65536;
static constexpr uint32_t numJobGrids = 256;
static constexpr uint32_t numJobSystemKernelThreads = 1024;
static constexpr uint32_t allActive = ~0u;

}

struct alignas(64) WaitData {
    char buf[16 * 1024 * 1024];
};

struct alignas(64) RunData {
    char buf[16 * 1024 * 1024];
};

struct JobGridInfo {
    uint32_t waitJobHead;
    uint32_t waitJobTail;
    uint32_t waitQueueLock;
    std::atomic_uint32_t numRunning;
    WaitData waitData;
    RunData runData;
};

static inline JobSystemConstants &jobSysConsts()
{
    return madronaTrainJobSysConstants;
}

static inline JobManager * getJobManager()
{
    return (JobManager *)jobSysConsts().jobSystemStateAddr;
}

static inline JobGridInfo *getGridInfo(JobManager *mgr)
{
    return (JobGridInfo *)((char *)mgr + jobSysConsts().jobGridsOffset);
}

static inline Job *getBaseJobList(JobManager *mgr)
{
    return (Job *)((char *)mgr + jobSysConsts().jobListOffset);
}

static void initializeJobSystem(JobManager *mgr)
{
    mgr->numOutstandingJobs = 0;

    for (int i = 0; i < 8; i++) {
        mgr->activeGrids[i] = 0;
    }

    auto grids = getGridInfo(mgr);

    for (int i = 0; i < (int)ICfg::numJobGrids; i++) {
        grids[i].waitJobHead = 0;
        grids[i].waitJobTail = 0;
        grids[i].waitQueueLock = 0;
        grids[i].numRunning = 0;
    }
}

static inline void freeJobData(void *data)
{
    free(data);
}

static inline bool isJobReady(Job &job)
{
    (void)job;
    return true;
}

static uint32_t warpInclusiveScan(uint32_t mask, uint32_t lane_id,
                                  uint32_t val)
{
#pragma unroll
    for (int i = 1; i < ICfg::numWarpThreads; i *= 2) {
        int tmp = __shfl_up_sync(mask, val, i);
        if ((int)lane_id >= i) {
            val += tmp;
        }
    }

    return val;
}

static inline uint32_t warpExclusiveScan(uint32_t mask, uint32_t lane_id,
                                         uint32_t val)
{
    return warpInclusiveScan(mask, lane_id, val) - val;
}

static inline uint32_t getHighestSetBit(uint32_t mask)
{
    return mask == 0u ? 0u : 31u - __clz(mask);
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
    uint32_t lane_id = threadIdx.x % ICfg::numWarpThreads;
    uint32_t warp_id = threadIdx.x / ICfg::numWarpThreads;

    constexpr uint32_t total_num_warps = ICfg::numJobSystemKernelThreads /
        ICfg::numWarpThreads;

    constexpr uint32_t grids_per_warp =
        ICfg::numJobGrids / total_num_warps;

    auto job_mgr = getJobManager();
    auto base_job_grids = getGridInfo(job_mgr);
    auto base_job_list = getBaseJobList(job_mgr);

    const uint32_t max_jobs_per_grid = jobSysConsts().maxJobsPerGrid;

    static __shared__ uint32_t active_grids_tmp[8];

    cuda::barrier<cuda::thread_scope_thread> cpy_barrier;
    init(&cpy_barrier, 1);

    auto findFirstReadyJob = [lane_id, max_jobs_per_grid](
        Job *job_list, uint32_t job_head, uint32_t job_tail) {
        int first_job_idx = -1;
        if (lane_id == 0) {
            for (uint32_t job_idx = job_head;
                 checkLTWrapped(job_idx, job_tail); job_idx++) {
                int wrapped_idx =
                    (int)wrapQueueIdx(job_idx, max_jobs_per_grid);
                Job &cur_job = job_list[wrapped_idx];

                if (isJobReady(cur_job)) {
                    first_job_idx = wrapped_idx;
                    break;
                }
            }
        }

        return __shfl_sync(ICfg::allActive, first_job_idx, 0);
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

        return __shfl_sync(ICfg::allActive, run_grid_idx, 0);
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
                wait_grid_offset * max_jobs_per_grid;

            uint32_t job_head = wait_grid.waitJobHead;
            // Cache the value of job tail, and use it across the warp,
            // it can be incremented by other threads
            uint32_t job_tail;
            if (lane_id == 0) {
                job_tail = wait_grid.waitJobTail;
            }
            job_tail = __shfl_sync(ICfg::allActive, job_tail, 0);

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
            auto isJobMergable = [first_ptr](Job &job) {
                return job.fn == first_ptr && isJobReady(job);
            };

            uint32_t num_bytes_per_job =
                waiting_jobs[first_job_idx].numBytesPerJob;

            uint32_t num_job_launches = 0;

            // Could start from the unwrapped version of first_job_idx,
            // but would need to change findFirstReadyJob to return
            // a separate failure boolean
            for (uint32_t job_offset = job_head;
                 /* Termination handled in loop */;
                 job_offset += ICfg::numWarpThreads) {
                uint32_t job_idx = job_offset + lane_id;

                bool inbounds = checkLTWrapped(job_idx, job_tail);
                uint32_t inbounds_mask =
                    __ballot_sync(ICfg::allActive, inbounds);

                // Warp is fully out of bounds
                if (inbounds_mask == 0) {
                    break;
                }

                // Note that this has to continue, so that the next iteration
                // of the loop (guaranteed to terminate) won't deadlock
                // due to the use of an all active mask
                if (!inbounds) {
                    continue;
                }

                Job cur_job = waiting_jobs[wrapQueueIdx(job_idx, max_jobs_per_grid)];

                bool merge_job = isJobMergable(cur_job);

                uint32_t merge_mask = __ballot_sync(inbounds_mask, merge_job);

                // Copy job data into grid's run buffer
                if (merge_job) {
                    uint32_t cur_num_launches = cur_job.numLaunches;
                    uint32_t num_prior_launches = num_job_launches +
                        warpExclusiveScan(merge_mask, lane_id,
                                          cur_num_launches);

                    cuda::memcpy_async(run_grid.runData.buf + 
                            num_prior_launches * num_bytes_per_job,
                        cur_job.data, cur_num_launches * num_bytes_per_job,
                        cpy_barrier);

                    num_job_launches = num_prior_launches + cur_num_launches;

                } 

                // Get current running total of jobs merged together for launch
                uint32_t top_merge_thread = getHighestSetBit(merge_mask);
                num_job_launches = __shfl_sync(inbounds_mask,
                    num_job_launches, top_merge_thread);
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
                 /* Termination handled in loop */;
                 job_offset -= ICfg::numWarpThreads) {

                uint32_t job_idx = job_offset - lane_id;

                bool inbounds = checkGEWrapped(job_offset, job_head);
                uint32_t inbounds_mask =
                    __ballot_sync(ICfg::allActive, inbounds);

                if (inbounds_mask == 0) {
                    break;
                }

                if (!inbounds) {
                    continue;
                }

                Job cur_job =
                    waiting_jobs[wrapQueueIdx(job_idx, max_jobs_per_grid)];

                bool mergable = isJobMergable(cur_job);

                if (mergable) {
                    freeJobData(cur_job.data);
                }

                // The sync here also ensures all threads are done
                // using cur_job before any pointers are overwritten
                // when coalescing the list
                uint32_t coalesce_mask =
                    __ballot_sync(inbounds_mask, !mergable);

                // Coalesce jobs that won't be launched to make the waiting
                // list contiguous
                if (!mergable) {
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

                uint32_t top_coalesce_thread = getHighestSetBit(coalesce_mask);
                base_wait_coalesce_idx = __shfl_sync(inbounds_mask,
                    base_wait_coalesce_idx, top_coalesce_thread);
            }

            if (lane_id == 0) {
                wait_grid.waitJobHead = base_wait_coalesce_idx;
            }

            std::atomic_thread_fence(std::memory_order_release);

            if (lane_id == 0) {
                uint32_t num_blocks = utils::divideRoundUp(num_job_launches,
                    ICfg::numJobLaunchKernelThreads);

                first_ptr<<<num_blocks, ICfg::numJobLaunchKernelThreads>>>(
                    (JobBase *)run_grid.runData.buf, num_job_launches,
                    run_grid_idx);
            }
        }

        __nanosleep(500);

        // Call this function at the end of the loop, in order to use
        // the same __syncthreads / acquire call for reading the activeGrid
        // bitfields and checking numOutstandingJobs
        nextLoopSetup();

        if (job_mgr->numOutstandingJobs == 0) {
            break;
        }
    }
}

static inline uint32_t computeMaxNumJobs(uint32_t num_worlds)
{
    // FIXME: scaling linearly like this probably doesn't make sense
    return ICfg::maxNumJobsPerWorld * num_worlds;
}

}

Context::WaveInfo Context::computeWaveInfo()
{
    using namespace gpuTrain;

    uint32_t active = __activemask();
    uint32_t num_active = __popc(active);

    uint32_t sel_idx = getHighestSetBit(active);
    uint32_t coalesced_idx = getNumLowerSetBits(active, lane_id_);

    return WaveInfo {
        .activeMask = active,
        .numActive = num_active,
        .leaderLane = sel_idx,
        .coalescedIDX = coalesced_idx,
    };
}

void * Context::allocJob(uint32_t total_bytes)
{
    return malloc(total_bytes);
}

void Context::queueJob(Job::EntryPtr func, gpuTrain::JobBase *data,
                       uint32_t num_launches, uint32_t num_bytes_per_job)
{
    using namespace gpuTrain;

    Job job {
        .fn = func,
        .data = data,
        .numLaunches = num_launches,
        .numBytesPerJob = num_bytes_per_job,
    };

    auto job_mgr = getJobManager();

    const auto base_job_grids = getGridInfo(job_mgr);
    const auto base_job_list = getBaseJobList(job_mgr);
    const uint32_t max_jobs_per_grid = jobSysConsts().maxJobsPerGrid;

    JobGridInfo &cur_grid = base_job_grids[grid_id_];
    Job *job_list = base_job_list + grid_id_ * max_jobs_per_grid;

    // Get lock
    std::atomic_thread_fence(std::memory_order_acquire);

    while (atomicCAS(&cur_grid.waitQueueLock, 0, 1) != 0) {}

    uint32_t cur_job_pos = cur_grid.waitJobTail;
    job_list[wrapQueueIdx(cur_job_pos, max_jobs_per_grid)] = job;

    cur_grid.waitJobTail++;

    cur_grid.waitQueueLock = 0;
    
    atomicAdd(&job_mgr->numOutstandingJobs, num_launches);

    std::atomic_thread_fence(std::memory_order_release);
}

void Context::markJobFinished(uint32_t num_jobs)
{
    using namespace gpuTrain;

    __syncthreads();

    if (threadIdx.x != 0) return;

    auto job_mgr = getJobManager();

    JobGridInfo &cur_grid = getGridInfo(job_mgr)[grid_id_];

    atomicSub(&job_mgr->numOutstandingJobs, num_jobs);

    uint32_t prev_running = cur_grid.numRunning.fetch_sub(num_jobs,
        std::memory_order_acq_rel);

    if (prev_running == num_jobs) {
        uint32_t grid_bitfield = grid_id_ / 32;
        uint32_t grid_bit = grid_id_ % 32;
        uint32_t mask = ~(1u << grid_bit);

        atomicAnd(&job_mgr->activeGrids[grid_bitfield], mask);
    }

    std::atomic_thread_fence(std::memory_order_release);
}

}

extern "C" __global__ void madronaTrainComputeJobSystemConstantsKernel(
    uint32_t num_worlds,
    madrona::gpuTrain::JobSystemConstants *out_constants,
    size_t *job_system_buffer_size)
{
    using namespace madrona;
    using namespace madrona::gpuTrain;

    uint32_t max_num_jobs_per_grid =
        madrona::gpuTrain::computeMaxNumJobs(num_worlds);

    size_t total_bytes = sizeof(JobManager);

    uint64_t grid_offset = utils::roundUp(total_bytes,
        std::alignment_of_v<JobGridInfo>);

    total_bytes = grid_offset + sizeof(JobGridInfo) * ICfg::numJobGrids;

    uint64_t wait_job_offset = madrona::utils::roundUp(total_bytes,
        std::alignment_of_v<Job>);

    uint64_t num_job_bytes = 
        sizeof(Job) * ICfg::numJobGrids * max_num_jobs_per_grid;

    total_bytes = wait_job_offset + num_job_bytes;

    *out_constants = JobSystemConstants {
        .jobSystemStateAddr = nullptr,
        .jobGridsOffset = (uint32_t)grid_offset,
        .jobListOffset = (uint32_t)wait_job_offset,
        .maxJobsPerGrid = max_num_jobs_per_grid,
    };

    *job_system_buffer_size = total_bytes;
}

extern "C" __global__  void madronaTrainInitializeJobSystemKernel()
{
    using namespace madrona;
    using namespace madrona::gpuTrain;

    auto job_mgr = getJobManager();
    new (job_mgr) JobManager();
    initializeJobSystem(job_mgr);
}

extern "C" __global__ void madronaTrainJobSystemKernel()
{
    madrona::gpuTrain::jobLoop();
}
