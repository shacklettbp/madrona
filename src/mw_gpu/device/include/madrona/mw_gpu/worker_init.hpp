#pragma once

#include <madrona/fwd.hpp>
#include <madrona/job.hpp>

#include <cstdint>

namespace madrona {

struct WorkerInit {
    JobID jobID;
    uint32_t gridID;
    uint32_t worldID;
    uint32_t laneID;
};

}
