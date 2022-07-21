#include <madrona/context.hpp>

#include "worker_init.hpp"

namespace madrona {

Context::Context(WorkerInit &&init)
    : job_mgr_(init.jobMgr),
      state_mgr_(init.stateMgr),
      io_mgr_(nullptr),
      worker_idx_(init.workerIdx)
{}

}
