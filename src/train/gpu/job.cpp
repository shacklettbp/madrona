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
static constexpr uint32_t numQueues = 128;
static constexpr uint32_t numWarpThreads = 32;
static constexpr uint32_t numJobSystemKernelThreads = 1024;

}

struct WaitQueueInfo {
    uint32_t jobHead;
    uint32_t jobTail;
    uint32_t numJobs;
};

static inline JobSystemConstants &jobSysConsts()
{
    return madronaTrainJobSysConstants;
}

static inline JobManager * getJobManager()
{
    return (JobManager *)jobSysConsts().jobSystemStateAddr;
}

static inline WaitQueueInfo *getWaitQueues(JobManager *mgr)
{
    return (WaitQueueInfo *)((char *)mgr + jobSysConsts().queueOffset);
}

static inline Job *getWaitingJobs(JobManager *mgr)
{
    return (Job *)((char *)mgr + jobSysConsts().jobsOffset);
}

static inline Job * getJobQueue(JobManager *mgr)
{
    return (Job *)((char *)mgr + jobSysConsts().queueOffset);
}

static inline char * getJobDataStorage(JobManager *mgr)
{
    return (char *)mgr + jobSysConsts().dataOffset;
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

// Only a single thread block can run this function
static void jobLoop()
{
    uint32_t lane_id = threadIdx.x % ICfg::numWarpThreads;
    uint32_t warp_id = threadIdx.x / ICfg::numWarpThreads;

    constexpr uint32_t total_num_warps = ICfg::numJobSystemKernelThreads /
        ICfg::numWarpThreads;

    constexpr uint32_t queues_per_warp =
        ICfg::numQueues / total_num_warps;

    const auto job_mgr = getJobManager();
    const auto base_wait_queues = getWaitQueues(job_mgr);
    const auto base_waiting_jobs = getWaitingJobs(job_mgr);
    const uint32_t num_jobs_per_queue = jobSysConsts().maxJobsPerQueue;

    while (true) {
        for (int queue_offset = 0; queue_offset < (int)queues_per_warp;
             queue_offset++) {
            uint32_t queue_idx = warp_id * queues_per_warp + queue_offset;

            WaitQueueInfo &wait_queue = base_wait_queues[queue_idx];
            Job *waiting_jobs =
                base_waiting_jobs + queue_offset * num_jobs_per_queue;

            int first_job_idx = -1;
            if (lane_id == 0) {
                for (int job_offset = 0; job_offset < (int)wait_queue.numJobs;
                     job_offset++) {
                    int job_idx = wait_queue.jobHead + job_offset;
                    Job &cur_job = waiting_jobs[job_idx];

                    if (isJobReady(cur_job)) {
                        first_job_idx = job_idx;
                        break;
                    }
                }
            }

            __syncwarp();

            first_job_idx = __shfl_sync(FULL_MASK, first_job_idx, 0);
            Job::EntryPtr first_ptr = waiting_jobs[first_job_idx].fn;

            for (int job_offset = 0; job_offset < (int)wait_queue.numJobs;
                 job_offset += ICfg::numWarpThreads) {
                int job_idx = first_job_idx + job_offset + lane_id;
                Job &cur_job = waiting_jobs[job_idx];

                if (isJobReady(cur_job) && cur_job.fn == first_ptr) {

                }
            }

            __syncwarp();

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
