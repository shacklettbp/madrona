#include "simple.hpp"

#include <madrona/mw_gpu.hpp>

namespace SimpleTaskgraph {

class GPUEntry : public madrona::GPUTaskgraphEntry<EnvInit, GPUEntry> {
public:
    static void init(const EnvInit *init, uint32_t num_worlds) {
        SimManager::init( init);
    };

    static void run(Engine &ctx) {
        SimpleSim::update(ctx);
    };
};

}
