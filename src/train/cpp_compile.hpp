#pragma once

#include <madrona/heap_array.hpp>

namespace madrona {
namespace cu {

HeapArray<char> compileToCUBIN(const char *code_path, int gpu_id,
                               const char **extra_options,
                               uint32_t num_extra_options);

}
}
