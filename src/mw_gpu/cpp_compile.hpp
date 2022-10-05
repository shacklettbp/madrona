/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
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
