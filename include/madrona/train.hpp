#pragma once

#include <cstdint>
#include <memory>

#include <madrona/span.hpp>

namespace madrona {

struct TrainConfig {
    uint32_t numWorlds;
    uint32_t gpuID;
};

struct CompileConfig {
    Span<const char * const> userSources;
    Span<const char * const> userCompileFlags;
    const char *entryFunc;
    const char *entryNamespace;
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
