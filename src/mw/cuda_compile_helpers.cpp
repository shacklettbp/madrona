#include "cuda_compile_helpers.hpp"

#include <madrona/crash.hpp>

namespace madrona::cu {

[[noreturn]] void nvrtcError(
        nvrtcResult err, const char *file,
        int line, const char *funcname) noexcept
{
    fatal(file, line, funcname, "%s", nvrtcGetErrorString(err));
}

[[noreturn]] void nvJitLinkError(
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
