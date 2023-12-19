/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include "cuda_compile_helpers.hpp"

#include <madrona/heap_array.hpp>

namespace madrona {
namespace cu {

struct CompileOutput {
    HeapArray<char> outputPTX;
    HeapArray<char> outputBinary;
};

CompileOutput jitCompileCPPSrc(const char *src,
                                const char *src_path,
                                const char **opt_compile_flags,
                                uint32_t num_opt_compile_flags,
                                const char **fast_compile_flags,
                                uint32_t num_fast_compile_flags,
                                bool nvvm_out);

CompileOutput jitCompileCPPFile(const char *src_path,
                                const char **compile_flags,
                                uint32_t num_compile_flags,
                                const char **fast_compile_flags,
                                uint32_t num_fast_compile_flags,
                                bool nvvm_out);

}
}
