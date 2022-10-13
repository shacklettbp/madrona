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
    uint32_t gpuID;
};

struct CompileConfig {
    const char *entryName;
    Span<const char * const> userSources;
    Span<const char * const> userCompileFlags;
    bool enableLTO;
};

class TrainingExecutor {
public:
    TrainingExecutor(const StateConfig &state_cfg,
                     const CompileConfig &compile_cfg);
    ~TrainingExecutor();

    void run();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
