#pragma once
#include <cuda_runtime.h>

namespace madrona {
namespace cu {

inline cudaError_t checkCuda(cudaError_t res, const char *msg,
                             bool fatal = true) noexcept;

#define STRINGIFY_HELPER(m) #m
#define STRINGIFY(m) STRINGIFY_HELPER(m)

#define LOC_APPEND(m) m ": " __FILE__ " @ " STRINGIFY(__LINE__)

#define REQ_CUDA(expr) checkCuda((expr), LOC_APPEND(#expr))
#define CHK_CUDA(expr) checkCuda((expr), LOC_APPEND(#expr), false)

inline void *allocGPU(size_t num_bytes);

inline void deallocGPU(void *ptr);

inline void *allocStaging(size_t num_bytes);

inline void *allocReadback(size_t num_bytes);

inline void deallocCPU(void *ptr);

inline void cpyCPUToGPU(cudaStream_t strm, void *gpu, void *cpu, size_t num_bytes);

inline void cpyGPUToCPU(cudaStream_t strm, void *cpu, void *gpu, size_t num_bytes);

inline cudaStream_t makeStream();

void printCudaError(cudaError_t res, const char *msg);

}

}
}

#include "cuda_utils.inl"
