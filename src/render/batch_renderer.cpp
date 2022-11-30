#include <madrona/batch_renderer.hpp>
#include <madrona/render.hpp>

#include <madrona/math.hpp>
#include <madrona/dyn_array.hpp>

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

struct ShaderState {
    PipelineShaders rt;
};

struct PipelineState {
    VkPipelineCache cache;
    VkPipelineLayout rtLayout;
    VkPipeline rt;
};

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
    ShaderState shaderState;
    PipelineState pipelineState;
    TLASData tlases;
    Assets cube;

    inline Impl(const Config &cfg);

    inline void render(const uint32_t *num_instances);
};

static ShaderState makeShaderState(const DeviceState &dev,
                                   const BatchRenderer::Config &cfg)
{
    using namespace std;
    std::vector<string> shader_defines(0);
    shader_defines.emplace_back(
        string("RES_X (") + to_string(cfg.renderWidth) + "u)");
    shader_defines.emplace_back(
        string("RES_Y (") + to_string(cfg.renderHeight) + "u)");

    uint32_t num_workgroups_x = divideRoundUp(cfg.renderWidth,
                                              VulkanConfig::localWorkgroupX);

    uint32_t num_workgroups_y = divideRoundUp(cfg.renderHeight,
                                              VulkanConfig::localWorkgroupY);

    shader_defines.emplace_back(
        string("NUM_WORKGROUPS_X (") + to_string(num_workgroups_x) + "u)");
    shader_defines.emplace_back(
        string("NUM_WORKGROUPS_Y (") + to_string(num_workgroups_y) + "u)");

    const char *shader_name = "basic.comp";

    PipelineShaders::initCompiler();

    PipelineShaders shader(dev,
        { std::string(shader_name) },
        { 
            BindingOverride { 0, 3, nullptr, cfg.numWorlds, 0 }, // TLAS
        },
        Span<const string>(shader_defines.data(), (CountT)shader_defines.size()),
        STRINGIFY(SHADER_DIR));

    return ShaderState {
        std::move(shader),
    };
}

static PipelineState makePipelineState(const DeviceState &dev,
                                       const ShaderState &shaders)
{
    // Pipeline cache (unsaved)
    VkPipelineCacheCreateInfo pcache_info {};
    pcache_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    VkPipelineCache pipeline_cache;
    REQ_VK(dev.dt.createPipelineCache(dev.hdl, &pcache_info, nullptr,
                                      &pipeline_cache));

    // Push constant
    VkPushConstantRange push_const {
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(RTPushConstant),
    };

    // Layout configuration
    std::array desc_layouts {
        shaders.rt.getLayout(0),
    };

    VkPipelineLayoutCreateInfo rt_layout_info;
    rt_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    rt_layout_info.pNext = nullptr;
    rt_layout_info.flags = 0;
    rt_layout_info.setLayoutCount =
        static_cast<uint32_t>(desc_layouts.size());
    rt_layout_info.pSetLayouts = desc_layouts.data();
    rt_layout_info.pushConstantRangeCount = 1;
    rt_layout_info.pPushConstantRanges = &push_const;

    VkPipelineLayout rt_layout;
    REQ_VK(dev.dt.createPipelineLayout(dev.hdl, &rt_layout_info, nullptr,
                                       &rt_layout));

    std::array<VkComputePipelineCreateInfo, 1> compute_infos;
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT subgroup_size;
    subgroup_size.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT;
    subgroup_size.pNext = nullptr;
    subgroup_size.requiredSubgroupSize = VulkanConfig::subgroupSize;

    compute_infos[0].sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compute_infos[0].pNext = nullptr;
    compute_infos[0].flags = 0;
    compute_infos[0].stage = {
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        &subgroup_size,
        VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT,
        VK_SHADER_STAGE_COMPUTE_BIT,
        shaders.rt.getShader(0),
        "main",
        nullptr,
    };
    compute_infos[0].layout = rt_layout;
    compute_infos[0].basePipelineHandle = VK_NULL_HANDLE;
    compute_infos[0].basePipelineIndex = -1;

    std::array<VkPipeline, compute_infos.size()> pipelines;
    REQ_VK(dev.dt.createComputePipelines(dev.hdl, pipeline_cache,
                                         compute_infos.size(),
                                         compute_infos.data(), nullptr,
                                         pipelines.data()));

    return PipelineState {
        pipeline_cache,
        rt_layout,
        pipelines[0],
    };
}

BatchRenderer::Impl::Impl(const Config &cfg)
    : inst(nullptr, enableValidation(), false, {}),
      dev(inst.makeDevice(getUUIDFromCudaID(cfg.gpuID), 1, 2, 1,
                          nullptr)),
      mem(dev, inst),
      blasAddrBuffer(mem.makeDedicatedBuffer(
              sizeof(uint64_t) * (uint64_t)cfg.maxObjects)),
      blasAddrBufferCUDA(dev, cfg.gpuID, blasAddrBuffer.mem,
          sizeof(uint64_t) * (uint64_t)cfg.maxObjects),
      renderQueue(makeQueue(dev, dev.computeQF, 0)),
      renderFence(makeFence(dev, false)),
      renderCmdPool(makeCmdPool(dev, dev.computeQF)),
      renderCmd(makeCmdBuffer(dev, renderCmdPool)),
      shaderState(makeShaderState(dev, cfg)),
      pipelineState(makePipelineState(dev, shaderState)),
      tlases(TLASData::setup(dev, GPURunUtil {
              renderCmdPool,
              renderCmd,  
              renderQueue,
              renderFence, 
          }, cfg.gpuID, mem, cfg.numWorlds, cfg.maxInstancesPerWorld)),
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

    dev.dt.cmdBindPipeline(renderCmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipelineState.rt);

    tlases.build(dev, num_instances, renderCmd);

    VkMemoryBarrier tlas_barrier;
    tlas_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    tlas_barrier.pNext = nullptr;
    tlas_barrier.srcAccessMask =
        VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    tlas_barrier.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT;

    dev.dt.cmdPipelineBarrier(renderCmd,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
        &tlas_barrier, 0, nullptr, 0, nullptr);

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

BatchRenderer::BatchRenderer(const Config &cfg)
    : impl_(nullptr)
{
    impl_ = std::unique_ptr<Impl>(new Impl(cfg));
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
