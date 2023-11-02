/*
 * Copyright 2021-2023 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once
#ifdef mw_gpu_EXPORTS
#define MADRONA_MWGPU_EXPORT MADRONA_EXPORT
#else
#define MADRONA_MWGPU_EXPORT MADRONA_IMPORT
#endif

#include <cstdint>
#include <memory>

#include <madrona/macros.hpp>
#include <madrona/span.hpp>
#include <madrona/importer.hpp>
#include <madrona/render/mw.hpp>

#include <cuda_runtime.h>
#include <cuda.h>

namespace madrona {

struct StateConfig {
    // worldInitPtr must be a pointer to numWorlds structs that are each
    // numWorldInitBytes in size. This must be a CPU pointer,
    // MWCudaExecutor will copy it to the GPU automatically.
    // These are equivalent to the MyPerWorldInit type referred to in
    // the mw_cpu.hpp documentation.
    void *worldInitPtr;
    uint32_t numWorldInitBytes;

    // CPU pointer to your application's config data. Equivalent to
    // MyConfig type in TaskGraphExecutor
    void *userConfigPtr;
    uint32_t numUserConfigBytes;

    // size and alignment of the per world global data type for the application
    uint32_t numWorldDataBytes; 
    uint32_t worldDataAlignment;

    // Batch size for the backend
    uint32_t numWorlds;

    // Number of exported ECS components
    uint32_t numExportedBuffers;
};

struct CompileConfig {
    enum class OptMode : uint32_t {
        Optimize,
        LTO,
        Debug,
    };

    // List of all the source files for your application that need to be
    // compiled on the GPU for the simulator to work correctly. These
    // will be combined with the core Madrona files and compiled by the
    // MWCudaExecutor constructor.
    Span<const char * const> userSources;
    // Any special flags (include directories, defines, etc) that need to be
    // passed to your source files
    Span<const char * const> userCompileFlags;

    // Explicitly set the GPU compilation mode. Recommend leaving as
    // OptMode::LTO, and using the MADRONA_MWGPU_FORCE_DEBUG=1 environment
    // variable to drop down to debug compilation when needed.
    OptMode optMode = OptMode::LTO;
};

class MWCudaExecutor {
public:
    // Initializes CUDA context, sets current device
    static CUcontext initCUDA(int gpu_id);

    MADRONA_MWGPU_EXPORT MWCudaExecutor(const StateConfig &state_cfg,
                                        const CompileConfig &compile_cfg,
                                        CUcontext cu_ctx);

    MADRONA_MWGPU_EXPORT MWCudaExecutor(MWCudaExecutor &&o);

    MADRONA_MWGPU_EXPORT ~MWCudaExecutor();

    // Run one invocation of the task graph across all worlds (one step)
    // Only returns after synchronization with the GPU is complete (not async)
    MADRONA_MWGPU_EXPORT void run();
    MADRONA_MWGPU_EXPORT void runAsync(cudaStream_t strm);

    // Get the base pointer of the component data exported with
    // ECSRegister::exportColumn. Note that this will be a GPU pointer.
    MADRONA_MWGPU_EXPORT void * getExported(CountT slot) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
