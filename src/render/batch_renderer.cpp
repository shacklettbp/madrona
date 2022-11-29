#include <madrona/batch_renderer.hpp>
#include <madrona/render.hpp>

#include <madrona/math.hpp>

#include "vk/core.hpp"
#include "vk/cuda_interop.hpp"
#include "vk/memory.hpp"
#include "vk/scene.hpp"
#include "vk/utils.hpp"

namespace madrona {
namespace render {

using namespace vk;

static bool enableValidation()
{
    char *validate_env = getenv("MADRONA_RENDER_VALIDATE");
    return validate_env  && validate_env[0] == '1';
}

struct BatchRenderer::Impl {
    InstanceState inst;
    DeviceState dev;
    MemoryAllocator mem;
    DedicatedBuffer blasAddrBuffer;
    CudaImportedBuffer blasAddrBufferCUDA;
    VkQueue renderQueue;
    VkFence renderFence;
    VkCommandPool renderCmdPool;
    VkCommandBuffer renderCmd;
    TLASData tlases;
    Assets cube;

    inline Impl(int64_t gpu_id,
                int64_t num_worlds,
                int64_t max_instances_per_world,
                int64_t max_objects);

    inline void render(const uint32_t *num_instances);
};

BatchRenderer::Impl::Impl(int64_t gpu_id,
                          int64_t num_worlds,
                          int64_t max_instances_per_world,
                          int64_t max_objects)
    : inst(nullptr, enableValidation(), false, {}),
      dev(inst.makeDevice(getUUIDFromCudaID(gpu_id), 1, 2, 1,
                                    nullptr)),
      mem(dev, inst),
      blasAddrBuffer(mem.makeDedicatedBuffer(sizeof(uint64_t) * max_objects)),
      blasAddrBufferCUDA(dev, gpu_id, blasAddrBuffer.mem,
          sizeof(uint64_t) * max_objects),
      renderQueue(makeQueue(dev, dev.computeQF, 0)),
      renderFence(makeFence(dev, false)),
      renderCmdPool(makeCmdPool(dev, dev.computeQF)),
      renderCmd(makeCmdBuffer(dev, renderCmdPool)),
      tlases(TLASData::setup(dev, GPURunUtil {
              renderCmdPool,
              renderCmd,  
              renderQueue,
              renderFence, 
          }, gpu_id, mem, num_worlds, max_instances_per_world)),
      cube(Assets::load(dev, mem))
{
    GPURunUtil gpu_run {
        renderCmdPool,
        renderCmd,
        renderQueue,
        renderFence,
    };

    uint64_t num_blas_addr_bytes =
        sizeof(uint64_t) * cube.blases.accelStructs.size();
    HostBuffer blas_addr_staging = mem.makeStagingBuffer(num_blas_addr_bytes);

    uint64_t *blas_addrs_staging_ptr = (uint64_t *)blas_addr_staging.ptr;
    for (int64_t i = 0; i < (int64_t)cube.blases.accelStructs.size(); i++) {
        blas_addrs_staging_ptr[i] = cube.blases.accelStructs[i].devAddr;
    }

    gpu_run.begin(dev);

    VkBufferCopy blas_addr_copy {
        .srcOffset = 0,
        .dstOffset = 0,
        .size = num_blas_addr_bytes,
    };

    dev.dt.cmdCopyBuffer(gpu_run.cmd,
                         blas_addr_staging.buffer,
                         blasAddrBuffer.buf.buffer,
                         1, &blas_addr_copy);

    gpu_run.submit(dev);
}

void BatchRenderer::Impl::render(const uint32_t *num_instances)
{
    REQ_VK(dev.dt.resetCommandPool(dev.hdl, renderCmdPool, 0));

    VkCommandBufferBeginInfo begin_info {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    REQ_VK(dev.dt.beginCommandBuffer(renderCmd, &begin_info));

    tlases.build(dev, num_instances, renderCmd);

    REQ_VK(dev.dt.endCommandBuffer(renderCmd));

    VkSubmitInfo submit_info {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.waitSemaphoreCount = 0;
    submit_info.pWaitSemaphores = nullptr;
    submit_info.pWaitDstStageMask = nullptr;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &renderCmd;
    REQ_VK(dev.dt.queueSubmit(renderQueue, 1, &submit_info, renderFence));

    waitForFenceInfinitely(dev, renderFence);
    resetFence(dev, renderFence);
}

BatchRenderer::BatchRenderer(int64_t gpu_id,
                             int64_t num_worlds,
                             int64_t max_instances_per_world,
                             int64_t max_objects)
    : impl_(nullptr)
{
    impl_ = std::unique_ptr<Impl>(
        new Impl(gpu_id, num_worlds, max_instances_per_world, max_objects));
}

BatchRenderer::BatchRenderer(BatchRenderer &&o)
    : impl_(std::move(o.impl_))
{}

BatchRenderer::~BatchRenderer() {}

AccelStructInstance ** BatchRenderer::tlasInstancePtrs() const
{
    return (AccelStructInstance **)
        impl_->tlases.instanceAddrsStorageCUDA.getDevicePointer();
}

uint64_t * BatchRenderer::objectsBLASPtr() const
{
    return (uint64_t *)
        impl_->blasAddrBufferCUDA.getDevicePointer();
}

void BatchRenderer::render(const uint32_t *num_instances)
{
    impl_->render(num_instances);
}

}
}
