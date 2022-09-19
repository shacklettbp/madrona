#include <madrona/context.hpp>

#include "worker_init.hpp"

namespace madrona {

Context::Context(WorkerInit &&init)
    : job_mgr_(init.jobMgr),
      state_mgr_(init.stateMgr),
      state_cache_(init.stateCache),
      io_mgr_(nullptr),
      worker_idx_(init.workerIdx),
      cur_job_id_(JobID::none())
#ifdef MADRONA_MW_MODE
      , cur_world_id_(0)
#endif
{}

}
