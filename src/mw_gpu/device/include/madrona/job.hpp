#pragma once

#include <array>
#include <atomic>
#include <cstdint>

namespace madrona {

struct JobID {
    uint32_t gen;
    uint32_t id;
};

// New types not used by the CPU implementation are hidden in the mwGPU
// namespace.
namespace mwGPU {
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
    using EntryPtr = void (*)(mwGPU::JobBase *,
                              uint32_t num_launches, uint32_t grid_id);
    EntryPtr fn;
    mwGPU::JobBase *data;
    uint32_t numInvocations;
    uint32_t numBytesPerJob;
};


class JobManager {
public:
    uint32_t numOutstandingJobs;
    uint32_t activeGrids[8];

    std::atomic<JobID> freeTrackerHead;
};

}

#include "job.inl"
