/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <madrona/fwd.hpp>

#include <cstdint>

namespace madrona {

struct WorkerInit {
#ifdef MADRONA_USE_JOB_SYSTEM
    JobManager *jobMgr;
    StateManager *stateMgr;
    StateCache *stateCache;
    int workerIdx;
#endif
#ifdef MADRONA_USE_TASK_GRAPH
    StateManager *stateMgr;
    StateCache *stateCache;
#endif
#ifdef MADRONA_MW_MODE
    uint32_t worldID;
#endif
};

}
