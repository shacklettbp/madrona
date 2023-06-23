#include <madrona/viz/recorder.hpp>

namespace madrona::viz {

struct Recorder::Impl {

    static Impl * init();
};

Recorder::Impl * Recorder::Impl::init()
{
}

Recorder::Recorder(const Config &cfg)
    : impl_(Impl::init())
{}

Recorder::Recorder(Recorder &&o) = default;
Recorder::~Recorder() = default;

const VizECSBridge * Recorder::rendererBridge() const
{
}

}
