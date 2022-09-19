#pragma once

#include <madrona/fwd.hpp>

namespace madrona {

struct WorkerInit {
    JobManager *jobMgr;
    StateManager *stateMgr;
    StateCache *stateCache;
    int workerIdx;
};

}
