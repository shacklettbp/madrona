/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include <madrona/cuda_utils.hpp>
#include <madrona/crash.hpp>

namespace madrona {
namespace cu {

[[noreturn]] MADRONA_EXPORT void cudaRuntimeError(
        cudaError_t err, const char *file,
        int line, const char *funcname) noexcept
{
    fatal(file, line, funcname, "%s", cudaGetErrorString(err));
}

[[noreturn]] MADRONA_EXPORT void cuDrvError(
        CUresult err, const char *file,
        int line, const char *funcname) noexcept
{
    const char *name, *desc;
    cuGetErrorName(err, &name);
    cuGetErrorString(err, &desc);
    fatal(file, line, funcname, "%s: %s", name, desc);
}

[[noreturn]] MADRONA_EXPORT void nvrtcError(
        nvrtcResult err, const char *file,
        int line, const char *funcname) noexcept
{
    fatal(file, line, funcname, "%s", nvrtcGetErrorString(err));
}

[[noreturn]] MADRONA_EXPORT void nvJitLinkError(
        nvJitLinkResult err, const char *file,
        int line, const char *funcname) noexcept
{
    const char *err_str;
    switch(err) {
    case NVJITLINK_SUCCESS: {
        FATAL("Passed NVJITLINK_SUCCESS to nvJitLinkError");
    } break;
    case NVJITLINK_ERROR_UNRECOGNIZED_OPTION: {
        err_str = "Unrecognized option";
    } break;
    case NVJITLINK_ERROR_MISSING_ARCH: {
        err_str = "Need to specify -arch=sm_NN";
    } break;
    case NVJITLINK_ERROR_INVALID_INPUT: {
        err_str = "Invalid Input";
    } break;
    case NVJITLINK_ERROR_PTX_COMPILE: {
        err_str = "PTX compilation error";
    } break;
    case NVJITLINK_ERROR_NVVM_COMPILE: {
        err_str = "NVVM compilation error";
    } break;
    case NVJITLINK_ERROR_INTERNAL: {
        err_str = "Internal error";
    } break;
    }

    fatal(file, line, funcname, "nvJitLink error: %s", err_str);
}

}
}
