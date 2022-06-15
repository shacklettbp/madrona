#pragma once

#include <cstdint>
#include <memory>

#include <madrona/span.hpp>

namespace madrona {

struct TrainConfig {
    uint32_t numWorlds;
    uint32_t gpuID;
};

class TrainingExecutor {
public:
    TrainingExecutor(const TrainConfig &cfg,
                     Span<const char *> user_cpp_files);
    ~TrainingExecutor();

    void run();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
