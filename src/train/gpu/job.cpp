#include <madrona/job.hpp>
#include <madrona/utils.hpp>

#include <cuda/barrier>

extern "C" {
__constant__ madrona::gpuTrain::JobSystemConstants madronaTrainJobSysConstants;
}

namespace madrona {
namespace gpuTrain {

namespace ICfg {

static constexpr uint32_t maxNumJobsPerWorld = 1024;
static constexpr uint32_t jobDataPerWorld = 65536;
static constexpr uint32_t numJobGrids = 256;
static constexpr uint32_t numWarpThreads = 32;
static constexpr uint32_t numJobSystemKernelThreads = 1024;

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
    uint32_t numWaitingJobs;
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
    mgr->numWaitingJobs = 0;
    mgr->numOutstandingJobs = 0;

    mgr->activeGrids.fill(0);
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
    return warpInclusiveScan(val) - val;
}

static inline uint32_t getHighestSetBit(uint32_t mask)
{
    return mask == 0u ? 0u : 31u - __clz(mask);
}

static inline uint32_t getNumLowerSetBits(uint32_t mask, uint32_t idx)
{
    mask >>= (32 - idx);

    return __popc(mask);
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

    const auto job_mgr = getJobManager();
    const auto base_job_grids = getGridInfo(job_mgr);
    const auto base_job_list = getBaseJobList(job_mgr);
    const uint32_t max_jobs_per_queue = jobSysConsts().maxJobsPerQueue;

    cuda::barrier<cuda::thread_scope_thread> cpy_barrier;
    init(&cpy_barrier, 1);

    auto findFirstReadyJob = [lane_id](JobGridInfo &run_grid, Job *job_list) {
        int first_job_idx = -1;
        if (lane_id == 0) {
            for (int job_offset = 0;
                 job_offset < (int)run_grid.numWaitingJobs;
                 job_offset++) {
                int job_idx = run_grid.waitJobHead + job_offset;
                Job &cur_job = job_list[job_idx];

                if (isJobReady(cur_job)) {
                    first_job_idx = job_idx;
                    break;
                }
            }
        }

        __syncwarp();

        return __shfl_sync(FULL_MASK, first_job_idx, 0);
    };

    while (true) {
        for (int grid_offset = 0; grid_offset < (int)grids_per_warp;
             grid_offset++) {
            uint32_t grid_idx = warp_id * grids_per_warp + grid_offset;

            // FIXME: check if grid is still active, if so, skip

            JobGridInfo &run_grid = base_job_grids[grid_idx];
            Job *waiting_jobs = base_job_list +
                grid_offset * max_jobs_per_queue;

            int first_job_idx = findFirstReadyJob(run_grid, waiting_jobs);

            if (first_job_idx == -1) {
                continue;
            }

            Job::EntryPtr first_ptr = waiting_jobs[first_job_idx].fn;
            uint32_t num_bytes_per_job =
                waiting_jobs[first_job_idx].numBytesPerJob;

            uint32_t num_job_launches = 0;
            uint32_t base_wait_coalesce_idx = first_job_idx;

            int num_mergeable_jobs = run_grid.numWaitingJobs - first_job_idx;

            for (int job_offset = 0; job_offset < num_mergeable_jobs;
                 job_offset += ICfg::numWarpThreads) {
                int job_idx = first_job_idx + job_offset + lane_id;
                Job cur_job = waiting_jobs[job_idx];

                bool merge_job = cur_job.fn == first_ptr && isJobReady(cur_job);

                uint32_t merge_mask = __ballot_sync(FULL_MASK, merge_job);
                uint32_t coalesce_mask = ~merge_mask;

                // Copy job data into grid's run buffer
                if (merge_job) {
                    uint32_t cur_num_launches = cur_job.numLaunches;
                    uint32_t num_prior_launches = num_job_launches +
                        warpExclusiveScan(merge_mask, lane_id,
                                          cur_num_launches);

                    cuda::memcpy_async(
                        dst + num_prior_launches * num_bytes_per_job,
                        cur_job.data, cur_num_launches * num_bytes_per_job,
                        cpy_barrier);

                    num_job_launches = num_prior_launches + cur_num_launches;

                } 

                // Get current running total of jobs merged together for launch
                __syncwarp();

                uint32_t top_merge_thread = getHighestSetBit(merge_mask);
                num_job_launches = __shfl_sync(FULL_MASK,
                    num_job_launches, top_merge_thread);

                // Coalesce jobs that won't be launched to make the waiting
                // list contiguous
                if (!merge_job) {
                    int coalesce_idx = base_wait_coalesce_idx +
                        getNumLowerSetBits(coalesce_mask, lane_id);

                    if (coalesce_idx != job_idx) {
                        waiting_jobs[coalesce_idx] = cur_job;
                    }

                    base_wait_coalesce_idx = coalesce_idx + 1;
                }

                uint32_t top_coalesce_thread = getHighestSetBit(coalesce_mask);
                base_wait_coalesce_idx = __shfl_sync(FULL_MASK,
                    base_wait_coalesce_idx, top_coalesce_thread);
            }

            // Wait for all the async copies
            cpy_barrier.arrive_and_wait();

            __syncwarp();

            // FIXME: synchronization with trying to queue into this grid?
            if (lane_id == 0) {
                run_grid.numWaitingJobs -= num_job_launches;
            }

        }

        uint32_t cur_num_jobs = job_mgr->numWaitingJobs;

        if (thread_pos < cur_num_jobs) {
            Job job = job_queue[job_mgr->jobHead + thread_pos];
            job.fn<<<1, 1>>>(this, job.data);
        }

        __syncthreads();

        if (thread_pos == 0) {
            atomicAdd(&job_mgr->jobHead, cur_num_jobs);
            atomicSub(&job_mgr->numWaitingJobs, cur_num_jobs);
            atomicAdd(&job_mgr->numOutstandingJobs, cur_num_jobs);
        }

        if (job_mgr->numOutstandingJobs == 0) {
            break;
        }
    }
}

void Context::queueJob(Job job)
{
    auto job_mgr = getJobManager();



    atomicAdd(&job_mgr->numWaitingJobs, 1u);
}

void Context::markJobFinished()
{
    auto job_mgr = getJobManager();
    atomicSub(&job_mgr->numOutstandingJobs, 1u);
}

}
}

extern "C" __global__ void madronaTrainComputeJobSystemConstants(
    uint32_t num_worlds, size_t *out_buffer_size, uint32_t *out_buffer
    madrona::gpuTrain::jobSysConstants *out_constants)
{
    uint32_t max_num_jobs = ICfg::maxNumJobsPerWorld * num_worlds;
    out_constants.maxNumJobs = max_num_jobs;
}

extern "C" __global__  void madronaTrainInitializeJobSystemKernel()
{
    auto job_mgr = madrona::gpuTrain::getJobManager();
    new (job_mgr) JobManager();
    madrona::gpUTrain::initializeJobSystem(job_mgr);
}

extern "C" __global__ void madronaTrainJobSystemKernel()
{
    madrona::gpuTrain::jobLoop()
}
