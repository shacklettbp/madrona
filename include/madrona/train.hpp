#pragma once

#include <cstdint>
#include <memory>

#include <madrona/span.hpp>

namespace madrona {

struct TrainConfig {
    uint32_t numWorlds;
    uint32_t numWorldDataBytes;
    uint32_t worldDataAlignment;
    uint32_t gpuID;
};

struct CompileConfig {
    const char *initFunc;
    const char *runFunc;
    const char *userNamespace;
    Span<const char * const> userSources;
    Span<const char * const> userCompileFlags;
};

class TrainingExecutor {
public:
    TrainingExecutor(const TrainConfig &train_cfg,
                     const CompileConfig &compile_cfg);
    ~TrainingExecutor();

    void run();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
