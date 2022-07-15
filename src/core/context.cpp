#include <madrona/context.hpp>

namespace madrona {

Context::Context(JobManager &job_mgr, StateManager &state_mgr, int worker_idx)
    : job_mgr_(&job_mgr),
      state_mgr_(&state_mgr),
      worker_idx_(worker_idx)
{}

}
