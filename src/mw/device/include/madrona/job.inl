#include <madrona/ecs.hpp>

#include <type_traits>

#include "mw_gpu/worker_init.hpp"
#include "mw_gpu/const.hpp"

namespace madrona {

namespace mwGPU {
namespace consts {

inline constexpr uint32_t numWarpThreads = 32;
inline constexpr uint32_t numMegakernelThreads = 256;
inline constexpr uint32_t numMegakernelWarps =
    numMegakernelThreads / numWarpThreads;

}
}

constexpr JobID JobID::none()
{
    return JobID {
        ~0u,
        ~0u,
    };
}

template <size_t N>
struct JobContainerBase::DepsArray {
    JobID dependencies[N];

    template <typename... DepTs>
    inline DepsArray(DepTs ...deps)
        : dependencies { deps ... }
    {}
};

template <> struct JobContainerBase::DepsArray<0> {
    template <typename... DepTs>
    inline DepsArray(DepTs...) {}
};

template <typename Fn, size_t N>
template <typename... DepTs>
JobContainer<Fn, N>::JobContainer(JobID job_id,
                                  uint32_t world_id,
                                  uint32_t num_invocations,
                                  Fn &&func,
                                  DepTs ...deps)
    : JobContainerBase {
          .jobID = job_id,
          .worldID = world_id,
          .numInvocations = num_invocations,
          .numDependencies = N,
      },
      dependencies(deps...),
      fn(std::forward<Fn>(func))
{}

JobManager * JobManager::get()
{
    return (JobManager *)GPUImplConsts::get().jobSystemAddr;
}

mwGPU::SharedJobTracker * JobManager::getSharedJobTrackers()
{
    return (mwGPU::SharedJobTracker *)((char *)this +
        mwGPU::GPUImplConsts::get().sharedJobTrackerOffset);
}

mwGPU::UserJobTracker * JobManager::getUserJobTrackers()
{
    return (mwGPU::UserJobTracker *)((char *)this +
        mwGPU::GPUImplConsts::get().userJobTrackerOffset);
}

template <typename ContextT>
ContextT JobManager::makeContext(JobID job_id, uint32_t grid_id,
                                 uint32_t world_id, uint32_t lane_id)
{
    using DataT = std::conditional_t<std::is_same_v<ContextT, Context>,
        WorldBase, typename ContextT::WorldDataT>;

    DataT *world_data;

    // If this is being called with the generic Context base class,
    // we need to look up the size of the world data in constant memory
    if constexpr (std::is_same_v<ContextT, Context>) {
        char *world_data_base = 
            (char *)mwGPU::GPUImplConsts::get().worldDataAddr;
        world_data = (DataT *)(world_data_base + world_id *
            mwGPU::GPUImplConsts::get().numWorldDataBytes);
    } else {
        DataT *world_data_base =
            (DataT *)mwGPU::GPUImplConsts::get().worldDataAddr;

        world_data = world_data_base + world_id;
    }

    return ContextT(world_data, WorkerInit {
        .jobID = job_id,
        .gridID = grid_id,
        .worldID = world_id,
        .laneID = lane_id,
    });
}

}
