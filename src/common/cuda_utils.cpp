/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include <madrona/cuda_utils.hpp>
#include <madrona/crash.hpp>

namespace madrona::cu {

[[noreturn]] void cudaRuntimeError(
        cudaError_t err, const char *file,
        int line, const char *funcname) noexcept
{
    fatal(file, line, funcname, "%s", cudaGetErrorString(err));
}

[[noreturn]] void cuDrvError(
        CUresult err, const char *file,
        int line, const char *funcname) noexcept
{
    const char *name, *desc;
    cuGetErrorName(err, &name);
    cuGetErrorString(err, &desc);
    fatal(file, line, funcname, "%s: %s", name, desc);
}

}
