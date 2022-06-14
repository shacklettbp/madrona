#include <cstdint>
#include <iostream>

#include "cuda_utils.hpp"
#include "job"

namespace madrona {

// Only a single thread block can run this function
__global__ void jobSystem(JobQueue *job_queue)
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

void initTraining()
{
    auto strm = cu::makeStream();

    int v = 5;
    JobQueue *job_queue = initJobQueue(strm, [v] __device__ (Context &ctx) {
        printf("Hi %d\n", v);
    });

    jobSystem<<<1, 1024, 0, strm>>>(job_queue);

    REQ_CUDA(cudaStreamSynchronize(strm));

    return 0;
}

}

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    madrona::initTraining();

    return 0;
}
