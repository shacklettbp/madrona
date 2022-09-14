#include <type_traits>

namespace madrona {

namespace mwGPU {

namespace ICfg {

static constexpr uint32_t numWarpThreads = 32;
static constexpr uint32_t numJobLaunchKernelThreads = 256;

}

JobBase::JobBase(uint32_t world_id, uint32_t job_id, uint32_t num_deps)
    : worldID(world_id),
      jobID(job_id),
      numDependencies(num_deps)
{}

template <size_t N>
template <typename... Args>
JobDependenciesBase<N>::JobDependenciesBase(uint32_t world_id, uint32_t job_id,
                                            Args && ...args)
    : JobBase(world_id, job_id, N),
      deps {
          std::forward<Args>(args)...
      }
{}

template <typename Fn, size_t N>
template <typename... Args>
JobContainer<Fn, N>::JobContainer(uint32_t world_id, uint32_t job_id, Fn &&func,
                                  Args && ...args)
    : JobDependenciesBase<N>(world_id, job_id, std::forward<Args>(args)...),
      fn(std::forward<Fn>(func))
{}

}

}
