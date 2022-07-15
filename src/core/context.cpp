#include <madrona/context.hpp>

namespace madrona {

Context::Context(JobManager &job_mgr, StateManager &state_mgr, void *world_data,
                 int worker_idx)
    : job_mgr_(&job_mgr),
      state_mgr_(&state_mgr),
      world_data_(world_data),
      worker_idx_(worker_idx)
{}

}
