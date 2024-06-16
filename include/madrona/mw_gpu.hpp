/*
 * Copyright 2021-2023 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <cstdint>
#include <memory>

#include <madrona/macros.hpp>
#include <madrona/span.hpp>
#include <madrona/importer.hpp>

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

    // Number of taskgraphs that will be constructed in the environment's setupTasks
    uint32_t numTaskGraphs;

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

struct RenderConfig {
    // Until we support custom rendering functions, for now we must toggle
    // between rendering color or depth.
    enum class RenderMode : uint32_t {
        Color,
        Depth,
        None
    };

    RenderMode renderMode;

    // Imported assets from disk.
    imp::ImportedAssets *importedAssets;

    // The raytracer output is square so the resolution of the outputs would be
    // renderResolution x renderResolution.
    uint32_t renderResolution;

    // Configure near and far planes of the rendering.
    float nearPlane;
    float farPlane;
};

class MWCudaExecutor;

class MWCudaLaunchGraph {
public:
    MWCudaLaunchGraph(MWCudaLaunchGraph &&);
    ~MWCudaLaunchGraph();

private:
    struct Impl;

    MWCudaLaunchGraph(Impl *impl);

    std::unique_ptr<Impl> impl_;

friend class MWCudaExecutor;
};

class MWCudaExecutor {
public:

    // Initializes CUDA context, sets current device
    static CUcontext initCUDA(int gpu_id);

    MWCudaExecutor(const StateConfig &state_cfg,
                   const CompileConfig &compile_cfg,
                   const RenderConfig &render_cfg,
                   CUcontext cu_ctx);

    MWCudaExecutor(MWCudaExecutor &&o);
    ~MWCudaExecutor();

    // Builds a CUDA graph that will launch all the taskgraphs specified by
    // taskgraph_ids one after the other. Typically this correspond to
    // one step across all worlds, or a subset of the logic for a step.
    template <EnumType EnumT>
    inline MWCudaLaunchGraph buildLaunchGraph(EnumT taskgraph_id);
    inline MWCudaLaunchGraph buildLaunchGraph(uint32_t taskgraph_id);
    MWCudaLaunchGraph buildLaunchGraph(Span<const uint32_t> taskgraph_ids);
    // Helper to build a a launch graph that launches all task graphs
    MWCudaLaunchGraph buildLaunchGraphAllTaskGraphs();

    // Builds the launch graph which launches the ray tracing
    MWCudaLaunchGraph buildRenderGraph();

    // Runs the pre-built CUDA graph stored in launch_graph synchronously
    void run(MWCudaLaunchGraph &launch_graph);
    // Runs the pre-built CUDA graph stored in launch_graph asynchronously on strm
    void runAsync(MWCudaLaunchGraph &launch_graph, cudaStream_t strm);

    // Get the base pointer of the component data exported with
    // ECSRegister::exportColumn. Note that this will be a GPU pointer.
    void * getExported(CountT slot) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}

#include "mw_gpu.inl"
