#include <madrona/job.hpp>

namespace madrona {

// Only a single thread block can run this function
void jobSystem(JobQueue *job_queue)
{
    uint32_t thread_pos = threadIdx.x;

    while (true) {
        uint32_t cur_num_jobs = job_queue->numWaitingJobs;

        if (thread_pos < cur_num_jobs) {
            Job job = job_queue->jobs[job_queue->jobHead + thread_pos];
            job.fn<<<1, 1>>>(job_queue, job.arg);
        }

        __syncthreads();

        if (thread_pos == 0) {
            atomicSub(&job_queue->numWaitingJobs, cur_num_jobs);
            atomicAdd(&job_queue->numOutstandingJobs, cur_num_jobs);
            atomicAdd(&job_queue->jobHead, cur_num_jobs);
        }

        if (job_queue->numOutstandingJobs == 0) {
            break;
        }
    }
}

}

extern "C" __global__ void jobSystemKernel(JobQueue *job_queue)
{
    madrona::jobSystem(job_queue);
}
