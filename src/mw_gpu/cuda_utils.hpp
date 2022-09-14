#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <madrona/macros.hpp>

namespace madrona {
namespace cu {

inline void *allocGPU(size_t num_bytes);

inline void deallocGPU(void *ptr);

inline void *allocStaging(size_t num_bytes);

inline void *allocReadback(size_t num_bytes);

inline void deallocCPU(void *ptr);

inline void cpyCPUToGPU(cudaStream_t strm, void *gpu, void *cpu, size_t num_bytes);

inline void cpyGPUToCPU(cudaStream_t strm, void *cpu, void *gpu, size_t num_bytes);

inline cudaStream_t makeStream();

[[noreturn]] void cudaError(cudaError_t err, const char *file,
                            int line, const char *funcname) noexcept;
[[noreturn]] void cuDrvError(CUresult err, const char *file,
                             int line, const char *funcname) noexcept;
[[noreturn]] void nvrtcError(nvrtcResult err, const char *file,
                             int line, const char *funcname) noexcept;

inline void checkCuda(cudaError_t res, const char *file,
                      int line, const char *funcname) noexcept;
inline void checkCuDrv(CUresult res, const char *file,
                       int line, const char *funcname) noexcept;
inline void checkNVRTC(nvrtcResult res, const char *file,
                       int line, const char *funcname) noexcept;

}
}

#define ERR_CUDA(err) madrona::cu::cudaError((err), __FILE__, __LINE__,\
                                             MADRONA_COMPILER_FUNCTION_NAME)
#define ERR_CU(err) madrona::cu::cuDrvError((err), __FILE__, __LINE__,\
                                            MADRONA_COMPILER_FUNCTION_NAME)
#define ERR_NVRTC(err) madrona::cu::nvrtcError((err), __FILE__, __LINE__,\
                                               MADRONA_COMPILER_FUNCTION_NAME)

#define REQ_CUDA(expr) madrona::cu::checkCuda((expr), __FILE__, __LINE__,\
                                              MADRONA_COMPILER_FUNCTION_NAME)
#define REQ_CU(expr) madrona::cu::checkCuDrv((expr), __FILE__, __LINE__,\
                                             MADRONA_COMPILER_FUNCTION_NAME)
#define REQ_NVRTC(expr) madrona::cu::checkNVRTC((expr), __FILE__, __LINE__,\
                                                MADRONA_COMPILER_FUNCTION_NAME)

#include "cuda_utils.inl"
