#include <cstdint>

#include "job.hpp"

void set_val(float *data, uint32_t idx, float v)
{
    data[idx] = v;
}

template <typename Fn>
__global__ void setInitialJobKernelAddress(JobQueue *job_queue)
{
    job_queue->jobs[0].fn = jobEntry<Fn>;
}

template <typename Fn>
JobQueue *initJobQueue(cudaStream_t strm, Fn &&fn)
{
    JobQueue *job_queue = (JobQueue *)allocGPU(sizeof(JobQueue));
    JobQueue *queue_staging = (JobQueue *)allocStaging(sizeof(JobQueue));

    queue_staging->jobHead = 0;
    queue_staging->numWaitingJobs = 1;
    queue_staging->numOutstandingJobs = 0;

    setInitialJobKernelAddress<Fn><<<1, 1, 0, strm>>>(queue_staging);

    queue_staging->jobs[0].arg = &job_queue->jobData.buffer;

    new (&(queue_staging->jobData.buffer)[0]) Fn(std::forward<Fn>(fn));

    cpyCPUToGPU(strm, job_queue, queue_staging, sizeof(JobQueue));
    REQ_CUDA(cudaStreamSynchronize(strm));

    deallocCPU(queue_staging);

    return job_queue;
}
