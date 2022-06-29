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

// New types not used by the CPU implementation are hidden in the gpuTrain
// namespace.
namespace gpuTrain {

struct JobSystemConstants {
    void *jobSystemStateAddr;
    uint32_t jobGridsOffset;
    uint32_t jobListOffset;
    uint32_t maxJobsPerGrid;
};

struct JobBase {
    inline JobBase(uint32_t world_id);

    uint32_t worldID;
};

template <typename Fn>
struct JobContainer : JobBase {
    inline JobContainer(uint32_t world_id, Fn &&func);

    [[no_unique_address]] Fn fn;
};

}

struct Job {
    using EntryPtr = void (*)(Context &, gpuTrain::JobBase *,
                              uint32_t num_launches, uint32_t grid_id);
    EntryPtr fn;
    gpuTrain::JobBase *data;
    uint32_t numLaunches;
    uint32_t numBytesPerJob;
};

class Context {
public:
    inline Context(uint32_t grid_id, uint32_t world_id, uint32_t lane_id);

    template <typename Fn>
    inline void queueJob(Fn &&fn);

private:
    struct WaveInfo {
        uint32_t activeMask;
        uint32_t numActive;
        uint32_t leaderLane;
        uint32_t coalescedIDX;
    };

    WaveInfo computeWaveInfo();

    void * allocJob(uint32_t total_bytes);

    void queueJob(Job::EntryPtr func, gpuTrain::JobBase *data,
                  uint32_t num_launches, uint32_t num_bytes_per_job);

    template <typename Fn>
    static CU_GLOBAL void jobEntry(gpuTrain::JobBase *job_data,
                                   uint32_t num_launches,
                                   uint32_t grid_id);

    void markJobFinished(uint32_t num_jobs);

    uint32_t grid_id_;
    uint32_t world_id_;
    uint32_t lane_id_;
};

class JobManager {
public:
    uint32_t numOutstandingJobs;

    std::array<uint32_t, 8> activeGrids;
};

}

#ifdef MADRONA_TRAIN_MODE
#include "job.inl"
#endif
