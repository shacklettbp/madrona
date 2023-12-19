/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include <madrona/macros.hpp>

namespace madrona::cu {

inline void *allocGPU(size_t num_bytes);

inline void deallocGPU(void *ptr);

inline void *allocStaging(size_t num_bytes);

inline void *allocReadback(size_t num_bytes);

inline void deallocCPU(void *ptr);

inline void cpyCPUToGPU(cudaStream_t strm, void *gpu, void *cpu, size_t num_bytes);

inline void cpyGPUToCPU(cudaStream_t strm, void *cpu, void *gpu, size_t num_bytes);

inline cudaStream_t makeStream();

[[noreturn]] void cudaRuntimeError(
        cudaError_t err, const char *file,
        int line, const char *funcname) noexcept;
[[noreturn]] void cuDrvError(
        CUresult err, const char *file,
        int line, const char *funcname) noexcept;

inline void checkCuda(cudaError_t res, const char *file,
                      int line, const char *funcname) noexcept;
inline void checkCuDrv(CUresult res, const char *file,
                       int line, const char *funcname) noexcept;

}

#define ERR_CUDA(err) ::madrona::cu::cudaError((err), __FILE__, __LINE__,\
                                               MADRONA_COMPILER_FUNCTION_NAME)
#define ERR_CU(err) ::madrona::cu::cuDrvError((err), __FILE__, __LINE__,\
                                              MADRONA_COMPILER_FUNCTION_NAME)

#define REQ_CUDA(expr) ::madrona::cu::checkCuda((expr), __FILE__, __LINE__,\
                                                MADRONA_COMPILER_FUNCTION_NAME)
#define REQ_CU(expr) ::madrona::cu::checkCuDrv((expr), __FILE__, __LINE__,\
                                               MADRONA_COMPILER_FUNCTION_NAME)

#include "cuda_utils.inl"
