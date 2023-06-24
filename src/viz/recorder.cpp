#include <madrona/viz/recorder.hpp>
#include "interop.hpp"

namespace madrona::viz {

struct Recorder::Impl {
    VizECSBridge bridge;

    static Impl * init(const Config &cfg);

    void record();
};

Recorder::Impl * Recorder::Impl::init(const Config &cfg)
{
}

Recorder::Recorder(const Config &cfg)
    : impl_(Impl::init(cfg))
{}

Recorder::Recorder(Recorder &&o) = default;
Recorder::~Recorder() = default;

const VizECSBridge * Recorder::rendererBridge() const
{
}

void Recorder::record()
{
    impl_->record();
}

}
