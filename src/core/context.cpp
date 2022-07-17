#include <madrona/context.hpp>

namespace madrona {

Context::Context(Context::Init &&init)
    : job_mgr_(init.jobMgr),
      state_mgr_(init.stateMgr),
      worker_idx_(init.workerIdx)
{}

}
