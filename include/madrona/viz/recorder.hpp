#pragma once

#include <madrona/exec_mode.hpp>
#include <madrona/viz/system.hpp>

namespace madrona::viz {

class Recorder {
public:
    struct Config {
        uint32_t renderWidth;
        uint32_t renderHeight;
        uint32_t numWorlds;
        uint32_t maxViewsPerWorld;
        uint32_t maxInstancesPerWorld;
        ExecMode execMode;
    };

    Recorder(const Config &cfg);
    Recorder(Recorder &&o);
    ~Recorder();

    const VizECSBridge * rendererBridge() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
