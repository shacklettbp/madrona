#include <madrona/batch_renderer.hpp>
#include <madrona/render.hpp>

#include <madrona/math.hpp>

#include "vk/core.hpp"
#include "vk/cuda_interop.hpp"
#include "vk/memory.hpp"
#include "vk/scene.hpp"

namespace madrona {
namespace render {

static bool enableValidation()
{
    char *validate_env = getenv("MADRONA_RENDER_VALIDATE");
    return validate_env  && validate_env[0] == '1';
}

struct BatchRenderer::Impl {
    cudaStream_t cpyStrm;
    vk::InstanceState inst;
    vk::DeviceState dev;
    vk::MemoryAllocator mem;
    void *o2wSrc;
    void *objIDsSrc;
    uint64_t numO2WBytes;
    uint64_t numObjIDsBytes;
    vk::DedicatedBuffer o2wCopyBuffer;
    vk::DedicatedBuffer objIDsCopyBuffer;
    vk::CudaImportedBuffer o2wCopyCuda;
    vk::CudaImportedBuffer objIDsCopyCuda;
    vk::TLASData tlases;
    vk::Assets cube;

    Impl(int64_t num_worlds,
         int64_t objs_per_world,
         void *o2w_cuda,
         void *obj_ids_cuda);

    inline void render();
};

BatchRenderer::Impl::Impl(int64_t num_worlds,
                          int64_t objs_per_world,
                          void *o2w_cuda,
                          void *obj_ids_cuda)
    : cpyStrm(),
      inst(nullptr, enableValidation(), false, {}),
      dev(inst.makeDevice(vk::getUUIDFromCudaID(0), 2, 2, 1, nullptr)),
      mem(dev, inst),
      o2wSrc(o2w_cuda),
      objIDsSrc(obj_ids_cuda),
      numO2WBytes(sizeof(ObjectToWorld) *
        (uint64_t)num_worlds * (uint64_t)objs_per_world),
      numObjIDsBytes(sizeof(ObjectID) *
        (uint64_t)num_worlds * (uint64_t)objs_per_world),
      o2wCopyBuffer(mem.makeDedicatedBuffer(numO2WBytes)),
      objIDsCopyBuffer(mem.makeDedicatedBuffer(numObjIDsBytes)),
      o2wCopyCuda(dev, 0, o2wCopyBuffer.mem, numO2WBytes),
      objIDsCopyCuda(dev, 0, objIDsCopyBuffer.mem, numObjIDsBytes),
      tlases(vk::TLASData::setup(dev, mem, num_worlds, objs_per_world)),
      cube(vk::Assets::load(dev, mem))
{
    cudaStreamCreate(&cpyStrm);
}

void BatchRenderer::Impl::render()
{
    cudaMemcpyAsync(o2wCopyCuda.getDevicePointer(),
                    o2wSrc,
                    numO2WBytes,
                    cudaMemcpyDeviceToDevice,
                    cpyStrm);

    cudaMemcpyAsync(objIDsCopyCuda.getDevicePointer(),
                    objIDsSrc,
                    numObjIDsBytes,
                    cudaMemcpyDeviceToDevice,
                    cpyStrm);

    cudaStreamSynchronize(cpyStrm);
}

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

void BatchRenderer::render()
{
    impl_->render();
}

}
}
