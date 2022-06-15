#include "cuda_utils.hpp"

#include <madrona/crash.hpp>

namespace madrona {
namespace cu {

[[noreturn]] void cudaError(cudaError_t err, const char *file,
                            int line) noexcept
{
    fatal(file, line, "%s", cudaGetErrorString(err));
}

[[noreturn]] void cuDrvError(CUresult err, const char *file,
                             int line) noexcept
{
    const char *name, *desc;
    cuGetErrorName(err, &name);
    cuGetErrorString(err, &desc);
    fatal(file, line, "%s: %s", name, desc);
}

[[noreturn]] void nvrtcError(nvrtcResult err, const char *file,
                             int line) noexcept
{
    fatal(file, line, "%s", nvrtcGetErrorString(err));
}

}
}
