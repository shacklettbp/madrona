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

#include <madrona/span.hpp>

namespace madrona {

struct StateConfig {
    void *worldInitPtr;
    uint32_t numWorldInitBytes;
    uint32_t numWorldDataBytes;
    uint32_t worldDataAlignment;
    uint32_t numWorlds;
    uint32_t numExportedBuffers;
    uint32_t gpuID;
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
    OptMode optMode = OptMode::Optimize;
    Executor execMode = Executor::TaskGraph;
};

class TrainingExecutor {
public:
    TrainingExecutor(const StateConfig &state_cfg,
                     const CompileConfig &compile_cfg);
    ~TrainingExecutor();

    void run();

    uint8_t * rgbObservations() const;
    float * depthObservations() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
