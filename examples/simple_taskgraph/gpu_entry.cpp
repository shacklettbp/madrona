#include "simple.hpp"

#include <madrona/mw_gpu.hpp>

namespace SimpleExample {

class GPUEntry : public madrona::GPUEntry<Engine, EnvInit, GPUEntry> {
public:
    static void init(Engine &ctx, const EnvInit &init) {
        SimpleSim::init(ctx, init);
    };

    static void run(Engine &ctx) {
        SimpleSim::update(ctx);
    };
};

}
