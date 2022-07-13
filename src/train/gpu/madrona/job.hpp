#pragma once

// This header needs to be included on the CPU side for initial setup reasons,
// so some special casing is required so the host compiler can read this file
#ifdef MADRONA_TRAIN_MODE
#include <madrona/fwd.hpp>
#include <madrona/span.hpp>

#include <cstdint>
#include <array>
#include <atomic>

#define CU_GLOBAL __global__
#else
#define CU_GLOBAL
#endif

namespace madrona {

struct JobID {
    uint32_t gen;
    uint32_t id;
};

// New types not used by the CPU implementation are hidden in the gpuTrain
// namespace.
namespace gpuTrain {
struct JobSystemConstants {
    void *jobSystemStateAddr;
    uint32_t jobGridsOffset;
    uint32_t jobListOffset;
    uint32_t maxJobsPerGrid;
    uint32_t jobTrackerOffset;
};

struct JobBase {
    inline JobBase(uint32_t world_id, uint32_t job_id, uint32_t num_deps);

    uint32_t worldID;
    uint32_t jobID;
    uint32_t numDependencies;
};

template <std::size_t N>
struct JobDependenciesBase : public JobBase {
    template <typename... Args>
    inline JobDependenciesBase(uint32_t world_id, uint32_t job_id,
                               Args && ...args);

    [[no_unique_address]] std::array<JobID, N> deps;
};

template <typename Fn, std::size_t N>
struct JobContainer : public JobDependenciesBase<N> {
    template <typename... Args>
    inline JobContainer(uint32_t world_id, uint32_t job_id, Fn &&func,
                        Args && ...args);

    [[no_unique_address]] Fn fn;
};

}

struct Job {
    using EntryPtr = void (*)(gpuTrain::JobBase *,
                              uint32_t num_launches, uint32_t grid_id);
    EntryPtr fn;
    gpuTrain::JobBase *data;
    uint32_t numInvocations;
    uint32_t numBytesPerJob;
};

class Context {
public:
    inline Context(uint32_t job_id, uint32_t grid_id, uint32_t world_id,
                   uint32_t lane_id);

    template <typename Fn, typename... Args>
    inline JobID queueJob(Fn &&fn, bool is_child = true,
                          Args && ...dependencies);

    template <typename Fn, typename... Args>
    inline JobID queueMultiJob(Fn &&fn, uint32_t num_invocations,
        bool is_child = true, Args && ...dependencies);

    void markJobFinished(uint32_t num_jobs);

private:
    struct WaveInfo {
        uint32_t activeMask;
        uint32_t numActive;
        uint32_t leaderLane;
        uint32_t coalescedIDX;
    };

    WaveInfo computeWaveInfo();

    JobID getNewJobID(bool link_parent);

    gpuTrain::JobBase * allocJob(uint32_t bytes_per_job, WaveInfo wave_info);

    void addToWaitList(Job::EntryPtr func, gpuTrain::JobBase *data,
                       uint32_t num_invocations, uint32_t num_bytes_per_job,
                       uint32_t lane_id, WaveInfo wave_info);

    uint32_t job_id_;
    uint32_t grid_id_;
    uint32_t world_id_;
    uint32_t lane_id_;
};

class JobManager {
public:
    uint32_t numOutstandingJobs;
    uint32_t activeGrids[8];

    std::atomic<JobID> freeTrackerHead;
};

}

#ifdef MADRONA_TRAIN_MODE
#include "job.inl"
#endif
