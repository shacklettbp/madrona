#pragma once

#include <madrona/macros.hpp>

#include <nvrtc.h>
#include <nvJitLink.h>

namespace madrona::cu {

[[noreturn]] void nvrtcError(
        nvrtcResult err, const char *file,
        int line, const char *funcname) noexcept;
[[noreturn]] void nvJitLinkError(
        nvJitLinkResult err, const char *file,
        int line, const char *funcname) noexcept;


inline void checkNVRTC(nvrtcResult res, const char *file,
                       int line, const char *funcname) noexcept;
inline void checknvJitLink(nvJitLinkResult res, const char *file,
                           int line, const char *funcname) noexcept;

}

#define ERR_NVRTC(err) \
    ::madrona::cu::nvrtcError((err), __FILE__, __LINE__,\
                              MADRONA_COMPILER_FUNCTION_NAME)
#define ERR_NVJITLINK(err) \
    ::madrona::cu::nvJitLinkError((err), __FILE__, __LINE__,\
                                  MADRONA_COMPILER_FUNCTION_NAME)

#define REQ_NVRTC(expr) \
    ::madrona::cu::checkNVRTC((expr), __FILE__, __LINE__,\
                              MADRONA_COMPILER_FUNCTION_NAME)
#define REQ_NVJITLINK(expr) \
    ::madrona::cu::checknvJitLink((expr), __FILE__, __LINE__,\
                                  MADRONA_COMPILER_FUNCTION_NAME)

#include "cuda_compile_helpers.inl"
