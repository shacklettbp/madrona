#pragma once

#include <madrona/job.hpp>
#include <madrona/state.hpp>

#include <madrona/mw_gpu/worker_init.hpp>

namespace madrona {

class Context {
public:
    inline Context(WorkerInit &&init);

    inline StateManager & state();

    template <typename Fn, typename... Deps>
    inline JobID submit(Fn &&fn, bool is_child = true,
                        Deps && ...dependencies);

    template <typename Fn, typename... Deps>
    inline JobID submitN(Fn &&fn, uint32_t num_invocations,
        bool is_child = true, Deps && ...dependencies);

    template <typename... ColTypes, typename Fn, typename... Deps>
    inline JobID forAll(const Query<ColTypes...> &query, Fn &&fn,
                        bool is_child = true,
                        Deps && ... dependencies);

    void markJobFinished(uint32_t num_jobs);

protected:
    template <typename ContextT, typename Fn, typename... Deps>
    inline JobID submitImpl(Fn &&fn, uint32_t num_invocations, bool is_child,
                            Deps && ... dependencies);

    template <typename ContextT, typename... ColTypes, typename Fn,
              typename... Deps>
    inline JobID forAllImpl(const Query<ColTypes...> &query, Fn &&fn,
                            bool is_child, Deps && ... dependencies);

private:
    struct WaveInfo {
        uint32_t activeMask;
        uint32_t numActive;
        uint32_t leaderLane;
        uint32_t coalescedIDX;
    };

    WaveInfo computeWaveInfo();

    JobID getNewJobID(bool link_parent);

    mwGPU::JobBase * allocJob(uint32_t bytes_per_job, WaveInfo wave_info);

    void addToWaitList(Job::EntryPtr func, mwGPU::JobBase *data,
                       uint32_t num_invocations, uint32_t num_bytes_per_job,
                       uint32_t lane_id, WaveInfo wave_info);

    uint32_t job_id_;
    uint32_t grid_id_;
    uint32_t world_id_;
    uint32_t lane_id_;
};

}

#include "context.inl"
