#pragma once

#include <madrona/job.hpp>
#include <madrona/state.hpp>

namespace madrona {

class Context {
public:
    inline Context(uint32_t job_id, uint32_t grid_id, uint32_t world_id,
                   uint32_t lane_id);

    inline StateManager & state();

    template <typename T>
    T & world();

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

}

#include "context.inl"
