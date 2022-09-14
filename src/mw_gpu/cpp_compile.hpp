#pragma once

#include <madrona/heap_array.hpp>

namespace madrona {
namespace cu {

HeapArray<char> compileSrcToCUBIN(const char *src,
                                  const char *src_path,
                                  const char **compile_flags,
                                  uint32_t num_compile_flags);

HeapArray<char> compileFileToCUBIN(const char *src_path,
                                   const char **compile_flags,
                                   uint32_t num_compile_flags);

}
}
