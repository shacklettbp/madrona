#pragma once
#include <cstdint>

namespace madrona {

struct JobQueue;

struct Job {
    using FuncPtr = void (*)(JobQueue *job_queue, void *);
    FuncPtr fn;
    void *arg;
};

struct alignas(64) JobData {
    char buffer[10 * 1024 * 1024];
};

struct JobQueue {
    Job jobs[16384];
    uint32_t jobHead;
    uint32_t numWaitingJobs;
    uint32_t numOutstandingJobs;
    JobData jobData;
};

class Context {
public:
    JobQueue &jobQueue;
};

template <typename Fn>
__global__ void jobEntry(JobQueue *job_queue, void *data);

}

#include "job.inl"
