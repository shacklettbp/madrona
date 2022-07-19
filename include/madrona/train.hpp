#pragma once

#include <cstdint>
#include <memory>

#include <madrona/span.hpp>

namespace madrona {

struct TrainConfig {
    uint32_t numWorlds;
    void *ctxData;
    uint32_t numCtxDataBytes;
    uint32_t ctxDataAlignment;
    uint32_t gpuID;
};

struct CompileConfig {
    const char *entryName;
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

// Not necessary on CPU-side currently
template <typename ContextT, typename BaseT> class TrainingEntry {};

}
