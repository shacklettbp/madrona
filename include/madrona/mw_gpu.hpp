/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
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

namespace madrona {

struct StateConfig {
    void *worldInitPtr;
    uint32_t numWorldInitBytes;
    void *userConfigPtr;
    uint32_t numUserConfigBytes;
    uint32_t numWorldDataBytes;
    uint32_t worldDataAlignment;
    uint32_t numWorlds;
    uint32_t numExportedBuffers;
    uint32_t gpuID;
};

struct CompileConfig {
    enum class OptMode : uint32_t {
        Optimize,
        LTO,
        Debug,
    };

    enum class Executor {
        JobSystem,
        TaskGraph,
    };

    const char *entryName;
    Span<const char * const> userSources;
    Span<const char * const> userCompileFlags;
    OptMode optMode = OptMode::LTO;
    Executor execMode = Executor::TaskGraph;
};

class MWCudaExecutor {
public:
    MADRONA_MWGPU_EXPORT MWCudaExecutor(const StateConfig &state_cfg,
                                        const CompileConfig &compile_cfg);

    MADRONA_MWGPU_EXPORT MWCudaExecutor(MWCudaExecutor &&o);

    MADRONA_MWGPU_EXPORT ~MWCudaExecutor();

    MADRONA_MWGPU_EXPORT void run();

    MADRONA_MWGPU_EXPORT void * getExported(CountT slot) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
