#pragma once
#include <madrona/fwd.hpp>

#include <cstdint>
#include <array>

// This header needs to be included on the CPU side for initial setup reasons,
// so some special casing is required so the host compiler can read this file
#ifdef MADRONA_TRAIN_MODE
#define CU_GLOBAL __global__
#else
#define CU_GLOBAL
#endif

namespace madrona {

struct Job {
    using EntryPtr = void (*)(Context &, void *);
    EntryPtr fn;
    void *data;
    uint32_t numLaunches;
    uint32_t numBytesPerJob;
};

// New types not used by the CPU implementation are hidden in the gpuTrain
// namespace.
namespace gpuTrain {

struct JobSystemConstants {
    void *jobSystemStateAddr;
    uint32_t jobGridsOffset;
    uint32_t jobListOffset;
    uint32_t maxJobsPerQueue;
};

}

class Context {
public:
    template <typename Fn>
    inline void queueJob(Fn &&fn);

    void queueJob(Job job);

private:
    template <typename Fn>
    static CU_GLOBAL void jobEntry(void *data, uint32_t num_launches,
                                   uint32_t grid_id);

    void markJobFinished();

    uint32_t grid_id_;
    uint32_t world_id_;
};

class JobManager {
public:
    uint32_t numWaitingJobs;
    uint32_t numOutstandingJobs;

    std::array<uint32_t, 8> activeGrids;
};

}

#ifdef MADRONA_TRAIN_MODE
#include "job.inl"
#endif
