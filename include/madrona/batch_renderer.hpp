#pragma once

#include <madrona/types.hpp>
#include <madrona/render.hpp>
#include <memory>

namespace madrona {
namespace render {

class BatchRenderer {
public:
    BatchRenderer(int64_t gpu_id,
                  int64_t num_worlds,
                  int64_t max_instances_per_world,
                  int64_t max_objects);
    BatchRenderer(BatchRenderer &&o);

    ~BatchRenderer();

    AccelStructInstance ** tlasInstancePtrs() const;

    uint64_t * objectsBLASPtr() const;

    void render(const uint32_t *num_instances);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
}
