/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include <madrona/context.hpp>

#include "worker_init.hpp"

namespace madrona {

Context::Context(WorldBase *world_data, const WorkerInit &init)
    : data_(world_data)
#ifdef MADRONA_USE_JOB_SYSTEM
      , job_mgr_(init.jobMgr),
      state_mgr_(init.stateMgr),
      state_cache_(init.stateCache),
      io_mgr_(nullptr),
      worker_idx_(init.workerIdx),
      cur_job_id_(JobID::none())
#endif
#ifdef MADRONA_USE_TASK_GRAPH
      , state_mgr_(init.stateMgr)
      , state_cache_(init.stateCache)
#endif
#ifdef MADRONA_MW_MODE
      , cur_world_id_(init.worldID)
#endif
{}

}
