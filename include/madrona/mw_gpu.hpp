/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
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

namespace madrona {

struct StateConfig {
    enum class CameraMode : uint32_t {
        Perspective,
        Lidar,
        None,
    };

    void *worldInitPtr;
    uint32_t numWorldInitBytes;
    uint32_t numWorldDataBytes;
    uint32_t worldDataAlignment;
    uint32_t numWorlds;
    uint32_t maxViewsPerWorld;
    uint32_t numExportedBuffers;
    uint32_t gpuID;
    CameraMode cameraMode;
    uint32_t renderWidth;
    uint32_t renderHeight;
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
    MADRONA_IMPORT MWCudaExecutor(const StateConfig &state_cfg,
                                  const CompileConfig &compile_cfg);

    MADRONA_IMPORT MWCudaExecutor(MWCudaExecutor &&o);

    MADRONA_IMPORT ~MWCudaExecutor();

    MADRONA_IMPORT CountT loadObjects(Span<const imp::SourceObject> objs);

    MADRONA_IMPORT void run();

    MADRONA_IMPORT uint8_t * rgbObservations() const;
    MADRONA_IMPORT float * depthObservations() const;

    MADRONA_IMPORT void * getExported(CountT slot) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
