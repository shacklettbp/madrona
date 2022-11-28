#include <madrona/batch_renderer.hpp>

#include "vk/core.hpp"
#include "vk/cuda_interop.hpp"

namespace madrona {
namespace render {

static bool enableValidation()
{
    char *validate_env = getenv("MADRONA_RENDER_VALIDATE");
    return validate_env  && validate_env[0] == '1';
}

struct BatchRenderer::Impl {
    vk::InstanceState inst;
    vk::DeviceState dev;

    Impl(CountT num_worlds, CountT objs_per_world,
         void *o2w_cuda, void *obj_ids_cuda)
        : inst(nullptr, enableValidation(), false, {}),
          dev(inst.makeDevice(vk::getUUIDFromCudaID(0), 2, 2, 1, nullptr))
    {
    }
};

BatchRenderer::BatchRenderer(CountT num_worlds,
                             CountT objs_per_world,
                             void *o2w_cuda,
                             void *obj_ids_cuda)
    : impl_(nullptr)
{
    impl_ = std::unique_ptr<Impl>(new Impl(num_worlds, objs_per_world,
                                           o2w_cuda, obj_ids_cuda));
}

BatchRenderer::BatchRenderer(BatchRenderer &&o)
    : impl_(std::move(o.impl_))
{}

BatchRenderer::~BatchRenderer() {}

}
}
