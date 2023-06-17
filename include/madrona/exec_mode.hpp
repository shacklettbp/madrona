#pragma once

#include <cstdint>

namespace madrona {

enum class ExecMode : uint32_t {
    CPU,
    CUDA,
};

}
