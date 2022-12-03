#include <madrona/batch_renderer.hpp>
#include <madrona/render.hpp>

#include <madrona/math.hpp>
#include <madrona/dyn_array.hpp>
#include <vulkan/vulkan_core.h>

#include "vk/core.hpp"
#include "vk/cuda_interop.hpp"
#include "vk/memory.hpp"
#include "vk/scene.hpp"
#include "vk/utils.hpp"
#include "vk/descriptors.hpp"
#include "vk/present.hpp"

namespace madrona {
namespace render {

using namespace vk;

static bool enableValidation()
{
    char *validate_env = getenv("MADRONA_RENDER_VALIDATE");
    return validate_env  && validate_env[0] == '1';
}

static bool forcePresent()
{
    char *force_present = getenv("MADRONA_RENDER_FORCE_PRESENT");
    return force_present && force_present[0] == '1';
}

struct RendererInit {
    bool validationEnabled;
    InstanceState inst;
    Optional<Window> presentWindow;
    Optional<VkSurfaceKHR> presentSurface;
};

struct ShaderState {
    PipelineShaders rt;
};

struct PipelineState {
    VkPipelineCache cache;
    VkPipelineLayout rtLayout;
    VkPipeline rt;
};

struct FramebufferState {
    DedicatedBuffer rgb;
    CudaImportedBuffer rgbCUDA;
    DedicatedBuffer depth;
    CudaImportedBuffer depthCUDA;
    uint32_t renderWidth;
    uint32_t renderHeight;
    uint32_t numViews;
};

struct AssetBuffers {
    LocalBuffer geometryBuffer;
};

struct DescriptorState {
    FixedDescriptorPool rtPool;
    VkDescriptorSet rtSet;
};

struct BatchRenderer::Impl {
    InstanceState inst;
    DeviceState dev;
    MemoryAllocator mem;
    Optional<PresentationState> presentState;
    DedicatedBuffer blasAddrBuffer;
    CudaImportedBuffer blasAddrBufferCUDA;
    VkQueue renderQueue;
    VkFence renderFence;
    VkCommandPool renderCmdPool;
    VkCommandBuffer renderCmd;
    ShaderState shaderState;
    PipelineState pipelineState;
    FramebufferState fb;
    AssetManager assetMgr;
    DedicatedBuffer viewDataBuffer;
    CudaImportedBuffer viewDataBufferCUDA;
    TLASData tlases;
    DescriptorState descriptors;
    VkSemaphore renderFinished;
    VkSemaphore swapchainReady;
    Assets cube;

    inline Impl(const Config &cfg);
    inline Impl(const Config &cfg, RendererInit &&init);

    inline void render(const uint32_t *num_instances);
};

static ShaderState makeShaderState(const DeviceState &dev,
                                   const BatchRenderer::Config &cfg,
                                   const RendererInit &init)
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

    if (init.validationEnabled) {
        shader_defines.emplace_back("VALIDATE");
    }

    const char *shader_name = "basic.comp";

    PipelineShaders::initCompiler();

    PipelineShaders shader(dev,
        { std::string(shader_name) },
        { 
            BindingOverride { 0, 1, nullptr, cfg.numViews, 0 }, // TLAS
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
        sizeof(shader::RTPushConstant),
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

static FramebufferState makeFramebuffer(const DeviceState &dev,
                                        MemoryAllocator &mem,
                                        const BatchRenderer::Config &cfg)
{
    uint64_t num_pixels = (uint64_t)cfg.renderHeight *
        (uint64_t)cfg.renderWidth * (uint64_t)cfg.numViews;

    uint64_t num_rgb_bytes = num_pixels * 4 * sizeof(uint8_t);
    uint64_t num_depth_bytes = num_pixels * sizeof(float);

    DedicatedBuffer rgb_buf = mem.makeDedicatedBuffer(num_rgb_bytes);
    CudaImportedBuffer rgb_buf_cuda(dev, cfg.gpuID, rgb_buf.mem,
                                    num_rgb_bytes);

    DedicatedBuffer depth_buf = mem.makeDedicatedBuffer(num_depth_bytes);
    CudaImportedBuffer depth_buf_cuda(dev, cfg.gpuID, depth_buf.mem,
                                    num_depth_bytes);

    return FramebufferState {
        std::move(rgb_buf),
        std::move(rgb_buf_cuda),
        std::move(depth_buf),
        std::move(depth_buf_cuda),
        cfg.renderWidth,
        cfg.renderHeight,
        cfg.numViews,
    };
}

static DescriptorState makeDescriptors(const DeviceState &dev,
                                       const BatchRenderer::Config &cfg,
                                       const ShaderState &shader_state,
                                       const FramebufferState &fb,
                                       const AssetManager &asset_mgr,
                                       const LocalBuffer &view_data_buffer,
                                       const TLASData &tlas_data)
{
    FixedDescriptorPool rt_pool(dev, shader_state.rt, 0, 1);
    VkDescriptorSet rt_set = rt_pool.makeSet();

    std::array<VkWriteDescriptorSet, 5> desc_updates;
    
    VkDescriptorBufferInfo view_data_info;
    view_data_info.buffer = view_data_buffer.buffer;
    view_data_info.offset = 0;
    view_data_info.range = VK_WHOLE_SIZE;

    DescHelper::storage(desc_updates[0],
                       rt_set, &view_data_info, 0);

    HeapArray<VkAccelerationStructureKHR> view_tlas_hdls(cfg.numViews);
    // FIXME
    assert(cfg.numViews == cfg.numWorlds);
    memcpy(view_tlas_hdls.data(),
           tlas_data.hdls.data(),
           cfg.numViews * sizeof(VkAccelerationStructureKHR));

    VkWriteDescriptorSetAccelerationStructureKHR as_update;
    DescHelper::accelStructs(desc_updates[1],
                             as_update,
                             rt_set, 
                             view_tlas_hdls.data(),
                             cfg.numViews,
                             1);

    VkDescriptorBufferInfo obj_data_info;
    obj_data_info.buffer = asset_mgr.addrBuffer.buf.buffer;
    obj_data_info.offset = sizeof(uint64_t) * asset_mgr.maxObjects;
    obj_data_info.range = sizeof(shader::ObjectData) * asset_mgr.maxObjects;

    DescHelper::storage(desc_updates[2],
                       rt_set, &obj_data_info, 2);

    VkDescriptorBufferInfo rgb_info;
    rgb_info.buffer = fb.rgb.buf.buffer;
    rgb_info.offset = 0;
    rgb_info.range = VK_WHOLE_SIZE;

    DescHelper::storage(desc_updates[3],
                       rt_set, &rgb_info, 3);

    VkDescriptorBufferInfo depth_info;
    depth_info.buffer = fb.depth.buf.buffer;
    depth_info.offset = 0;
    depth_info.range = VK_WHOLE_SIZE;

    DescHelper::storage(desc_updates[4],
                       rt_set, &depth_info, 4);

    DescHelper::update(dev, desc_updates.data(), desc_updates.size());

    return DescriptorState {
        std::move(rt_pool),
        rt_set,
    };
}

static RendererInit setupRendererInit(const BatchRenderer::Config &cfg)
{
    bool validate = enableValidation();
    bool present = forcePresent();

    PFN_vkGetInstanceProcAddr get_instance = nullptr;

    auto present_window = Optional<Window>::none();
    if (present) {
        get_instance = PresentationState::init();
        present_window =
            PresentationState::makeWindow(cfg.renderWidth, cfg.renderHeight);
    }

    InstanceState inst(get_instance, validate,
          present_window.has_value(), present_window.has_value() ? 
              PresentationState::getInstanceExtensions(*present_window) :
              HeapArray<const char *>(0));

    auto present_surface = Optional<VkSurfaceKHR>::none();
    if (present) {
        present_surface =
            PresentationState::makeSurface(inst, *present_window);
    }

    return {
        validate,
        std::move(inst),
        present_window,
        present_surface,
    };
}

BatchRenderer::Impl::Impl(const Config &cfg)
    : Impl(cfg, setupRendererInit(cfg))
{}

BatchRenderer::Impl::Impl(const Config &cfg, RendererInit &&init)
    : inst(std::move(init.inst)),
      dev(inst.makeDevice(getUUIDFromCudaID(cfg.gpuID), 1, 2, 1,
                          init.presentSurface)),
      mem(dev, inst),
      presentState(init.presentWindow.has_value() ?
          PresentationState(inst, dev, std::move(*init.presentWindow),
                            std::move(*init.presentSurface),
                            dev.computeQF, 1, true) :
          Optional<PresentationState>::none()),
      blasAddrBuffer(mem.makeDedicatedBuffer(
          sizeof(uint64_t) * (uint64_t)cfg.maxObjects)),
      blasAddrBufferCUDA(dev, cfg.gpuID, blasAddrBuffer.mem,
          sizeof(uint64_t) * (uint64_t)cfg.maxObjects),
      renderQueue(makeQueue(dev, dev.computeQF, 0)),
      renderFence(makeFence(dev, false)),
      renderCmdPool(makeCmdPool(dev, dev.computeQF)),
      renderCmd(makeCmdBuffer(dev, renderCmdPool)),
      shaderState(makeShaderState(dev, cfg, init)),
      pipelineState(makePipelineState(dev, shaderState)),
      fb(makeFramebuffer(dev, mem, cfg)),
      assetMgr(dev, mem, cfg.gpuID, cfg.maxObjects),
      viewDataBuffer(mem.makeDedicatedBuffer(
          sizeof(shader::ViewData) * cfg.numViews)),
      viewDataBufferCUDA(dev, cfg.gpuID, viewDataBuffer.mem,
          sizeof(shader::ViewData) * cfg.numViews),
      tlases(TLASData::setup(dev, GPURunUtil {
              renderCmdPool,
              renderCmd,  
              renderQueue,
              renderFence, 
          }, cfg.gpuID, mem, cfg.numWorlds, cfg.maxInstancesPerWorld)),
      descriptors(makeDescriptors(dev, cfg, shaderState, fb, assetMgr,
                                  viewDataBuffer.buf, tlases)),
      renderFinished(presentState.has_value() ? makeBinarySemaphore(dev) :
                     VK_NULL_HANDLE),
      swapchainReady(presentState.has_value() ? makeBinarySemaphore(dev) :
                     VK_NULL_HANDLE),
      cube(assetMgr.loadCube(dev, mem))
{
    if (presentState.has_value()) {
        GPURunUtil tmp_run {
            renderCmdPool,
            renderCmd,
            renderQueue,
            renderFence,
        };
        presentState->forceTransition(dev, tmp_run);
    }
}

void BatchRenderer::Impl::render(const uint32_t *num_instances)
{
    REQ_VK(dev.dt.resetCommandPool(dev.hdl, renderCmdPool, 0));

    VkCommandBufferBeginInfo begin_info {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    REQ_VK(dev.dt.beginCommandBuffer(renderCmd, &begin_info));

    dev.dt.cmdBindPipeline(renderCmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipelineState.rt);
    dev.dt.cmdBindDescriptorSets(renderCmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 pipelineState.rtLayout, 0, 1,
                                 &descriptors.rtSet, 0, nullptr);

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

    dev.dt.cmdDispatch(
        renderCmd,
        fb.renderWidth,
        fb.renderHeight,
        fb.numViews);

    REQ_VK(dev.dt.endCommandBuffer(renderCmd));

    VkSubmitInfo submit_info {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.waitSemaphoreCount = 0;
    submit_info.pWaitSemaphores = nullptr;
    submit_info.pWaitDstStageMask = nullptr;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &renderCmd;

    uint32_t swapchain_idx = 0;
    VkPipelineStageFlags present_wait_mask =
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    if (presentState.has_value()) {
        swapchain_idx = presentState->acquireNext(dev, swapchainReady);
        submit_info.waitSemaphoreCount = 1;
        submit_info.pWaitSemaphores = &swapchainReady;

        submit_info.pWaitDstStageMask = &present_wait_mask;

        submit_info.pSignalSemaphores = &renderFinished;
        submit_info.signalSemaphoreCount = 1;
    }

    REQ_VK(dev.dt.queueSubmit(renderQueue, 1, &submit_info, renderFence));

    if (presentState.has_value()) {
        presentState->present(dev, swapchain_idx, renderQueue,
                              1, &renderFinished);
    }

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

void * BatchRenderer::viewDataPtr() const
{
    return
        impl_->viewDataBufferCUDA.getDevicePointer();
}

uint8_t * BatchRenderer::rgbPtr() const
{
    return (uint8_t *)
        impl_->fb.rgbCUDA.getDevicePointer();
}

float * BatchRenderer::depthPtr() const
{
    return (float *)
        impl_->fb.depthCUDA.getDevicePointer();
}

void BatchRenderer::render(const uint32_t *num_instances)
{
    impl_->render(num_instances);
}

}
}
