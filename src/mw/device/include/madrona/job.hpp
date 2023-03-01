/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <array>
#include <atomic>
#include <cstdint>

namespace madrona {

struct JobID {
    uint32_t gen;
    uint32_t id;

    static constexpr inline JobID none();
};

struct JobContainerBase {
    JobID jobID;
    uint32_t worldID;
    uint32_t numInvocations;
    uint32_t numDependencies;

    template <size_t N> struct DepsArray;
};

template <typename Fn, size_t N>
struct JobContainer : public JobContainerBase {
    [[no_unique_address]] DepsArray<N> dependencies;
    [[no_unique_address]] Fn fn;

    template <typename... DepTs>
    inline JobContainer(JobID job_id, uint32_t world_id,
                        uint32_t num_invocations, Fn &&fn, DepTs ...deps);
};

struct Job {
    JobContainerBase *data;
    uint32_t funcID;
    uint32_t numCombinedJobs;
    uint32_t numBytesPerJob;
};

class JobManager {
public:
    uint32_t numOutstandingInvocations;

    AtomicU32<JobID> freeTrackerHead;

    bool startBlockIter(RunnableJob *out_job);
    void finishBlockIter();

    static inline JobManager * get();

    template <typename ContextT>
    static inline ContextT makeContext(JobID job_id, uint32_t grid_id,
                                       uint32_t world_id, uint32_t lane_id);
};

}

#include "job.inl"
