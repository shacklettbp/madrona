#pragma once

#include <madrona/types.hpp>
#include <memory>

namespace madrona {
namespace render {

class BatchRenderer {
public:
    BatchRenderer(CountT num_worlds,
                  CountT objs_per_world,
                  void *o2w_cuda,
                  void *obj_ids_cuda);
    BatchRenderer(BatchRenderer &&o);

    ~BatchRenderer();

    void render();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
}
