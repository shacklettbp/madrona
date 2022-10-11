#include "simple.hpp"

#include <madrona/mw_gpu.hpp>

namespace SimpleExample {

class GPUEntry : public madrona::GPUEntry<Engine, GPUEntry> {
public:
    static void init(Engine &ctx) {
        SimpleSim::init(ctx);
    };

    static void run(Engine &ctx) {
        SimpleSim::update(ctx);
    };
};

}
