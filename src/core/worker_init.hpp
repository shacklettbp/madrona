#pragma once

#include <madrona/fwd.hpp>

namespace madrona {

struct WorkerInit {
    JobManager *jobMgr;
    StateManager *stateMgr;
    StateCache *stateCache;
    int workerIdx;
#ifdef MADRONA_MW_MODE
    uint32_t worldID;
#endif
};

}
