#pragma once

#include <madrona/job.hpp>

namespace madrona {
namespace mw {

struct JobContainerWorldBase : public madrona::JobContainerBase {
    uint32_t worldID;

    inline JobContainerWorldBase(uint32_t world_id, uint32_t num_deps)
        : madrona::JobContainerBase {
              .id = JobID::none(), // Assigned by JobManager
              .numDependencies = num_deps,
          },
          worldID(world_id)
    {}
};

template <typename Fn, size_t N>
struct JobContainerWorld : public JobContainerWorldBase {
    [[no_unique_address]] DepsArray<N> dependencies;
    [[no_unique_address]] Fn fn;

    template <typename... DepTs>
    inline JobContainerWorld(uint32_t world_id, Fn &&fn, DepTs ...deps)
        : JobContainerWorldBase(world_id, N),
          dependencies(deps...),
          fn(std::forward<Fn>(fn))
    {}
};

}
}
