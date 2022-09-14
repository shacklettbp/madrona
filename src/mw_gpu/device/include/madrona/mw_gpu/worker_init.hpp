#pragma once

#include <madrona/fwd.hpp>

#include <cstdint>

namespace madrona {

struct WorkerInit {
    uint32_t jobID;
    uint32_t gridID;
    uint32_t worldID;
    uint32_t laneID;
};

}
