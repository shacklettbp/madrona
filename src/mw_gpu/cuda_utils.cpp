#include "cuda_utils.hpp"

#include <madrona/crash.hpp>

namespace madrona {
namespace cu {

[[noreturn]] void cudaError(cudaError_t err, const char *file,
                            int line, const char *funcname) noexcept
{
    fatal(file, line, funcname, "%s", cudaGetErrorString(err));
}

[[noreturn]] void cuDrvError(CUresult err, const char *file,
                             int line, const char *funcname) noexcept
{
    const char *name, *desc;
    cuGetErrorName(err, &name);
    cuGetErrorString(err, &desc);
    fatal(file, line, funcname, "%s: %s", name, desc);
}

[[noreturn]] void nvrtcError(nvrtcResult err, const char *file,
                             int line, const char *funcname) noexcept
{
    fatal(file, line, funcname, "%s", nvrtcGetErrorString(err));
}

}
}
