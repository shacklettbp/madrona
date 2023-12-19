#include "../batch_renderer.hpp"
#include "scene.hpp"

#include <madrona/math.hpp>
#include <madrona/dyn_array.hpp>
#include <madrona/tracing.hpp>
#include <madrona/render/vk/backend.hpp>

#include <filesystem>

#include "vk/cuda_interop.hpp"
#include "vk/memory.hpp"
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

static bool debugPresent()
{
    char *debug_present = getenv("MADRONA_RENDER_DEBUG_PRESENT");
    return debug_present && debug_present[0] == '1';
}

struct ImplInit {
    bool validationEnabled;
    Backend backend;
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
    Backend backend;
    Device dev;
    MemoryAllocator mem;
    Optional<PresentationState> presentState;
    Optional<LocalImage> rgbPresentIntermediate;
    VkQueue renderQueue;
    VkFence renderFence;
    VkCommandPool renderCmdPool;
    VkCommandBuffer renderCmd;
    ShaderState shaderState;
    PipelineState pipelineState;
    FramebufferState fb;
    AssetManager assetMgr;
    EngineToRendererBuffer viewDataBuffer;
    HostToEngineBuffer viewDataAddrsBuffer;
    TLASData tlases;
    DescriptorState descriptors;
    uint32_t launchWidth;
    uint32_t launchHeight;
    VkSemaphore renderFinished;
    VkSemaphore swapchainReady;
    DynArray<Assets> loadedAssets;

    inline Impl(const Config &cfg);
    inline Impl(const Config &cfg, ImplInit &&init);
    inline ~Impl();
   
    inline CountT loadObjects(Span<const imp::SourceObject> objs);

    inline void render();
};

static ShaderState makeShaderState(const Device &dev,
                                   const BatchRenderer::Config &cfg,
                                   const ImplInit &init)
{
    using namespace std;
    DynArray<pair<string, Optional<string>>> shader_defines(0);

    shader_defines.emplace_back("RES_X", 
        string("(") + to_string(cfg.renderWidth) + "u)");

    shader_defines.emplace_back("RES_Y",
        string("(") + to_string(cfg.renderHeight) + "u)");

    shader_defines.emplace_back("MAX_VIEWS_PER_WORLD",
        string("(") + to_string(cfg.maxViewsPerWorld) + "u)");

    uint32_t num_workgroups_x = divideRoundUp(cfg.renderWidth,
                                              VulkanConfig::localWorkgroupX);

    uint32_t num_workgroups_y = divideRoundUp(cfg.renderHeight,
                                              VulkanConfig::localWorkgroupY);

    shader_defines.emplace_back("NUM_WORKGROUPS_X",
        string("(") + to_string(num_workgroups_x) + "u)");

    shader_defines.emplace_back("NUM_WORKGROUPS_Y",
        string("(") + to_string(num_workgroups_y) + "u)");

    if (init.validationEnabled) {
        shader_defines.emplace_back("VALIDATE", Optional<string>::none());
    }

    if (cfg.cameraMode == render::CameraMode::Lidar) {
        shader_defines.emplace_back("LIDAR", Optional<string>::none());
    } else if (cfg.cameraMode == render::CameraMode::Perspective) {
        shader_defines.emplace_back("PERSPECTIVE", Optional<string>::none());
    }

    HeapArray<ShaderCompiler::MacroDefn> shader_macro_defns(
        shader_defines.size());
    for (CountT i = 0; i < shader_macro_defns.size(); i++) {
        const auto &defn = shader_defines[i];
        shader_macro_defns[i].name = defn.first.c_str();
        shader_macro_defns[i].value = defn.second.has_value() ?
            defn.second->c_str() : nullptr;
    }

    const char *shader_name = "basic.hlsl";

    ShaderCompiler compiler;
    SPIRVShader spirv = compiler.compileHLSLFileToSPV(
        (std::filesystem::path(STRINGIFY(SHADER_DIR)) / shader_name).c_str(),
        {}, shader_macro_defns);

    PipelineShaders shader(dev, spirv, {});

    return ShaderState {
        std::move(shader),
    };
}

static PipelineState makePipelineState(const Device &dev,
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
        "render",
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

static FramebufferState makeFramebuffer(const Device &dev,
                                        MemoryAllocator &mem,
                                        const BatchRenderer::Config &cfg)
{
    CountT max_num_views =
        (CountT)cfg.numWorlds * (CountT)cfg.maxViewsPerWorld;

    uint64_t num_pixels = (uint64_t)cfg.renderHeight *
        (uint64_t)cfg.renderWidth * (uint64_t)max_num_views;

    uint64_t num_rgb_bytes = num_pixels * 4 * sizeof(uint8_t);
    uint64_t num_depth_bytes = num_pixels * sizeof(float);

    DedicatedBuffer rgb_buf =
        mem.makeDedicatedBuffer(num_rgb_bytes, false, true);
    CudaImportedBuffer rgb_buf_cuda(dev, cfg.gpuID, rgb_buf.mem,
                                    num_rgb_bytes);

    DedicatedBuffer depth_buf =
        mem.makeDedicatedBuffer(num_depth_bytes, false, true);
    CudaImportedBuffer depth_buf_cuda(dev, cfg.gpuID, depth_buf.mem,
                                    num_depth_bytes);

    return FramebufferState {
        std::move(rgb_buf),
        std::move(rgb_buf_cuda),
        std::move(depth_buf),
        std::move(depth_buf_cuda),
        cfg.renderWidth,
        cfg.renderHeight,
        uint32_t(max_num_views),
    };
}

static DescriptorState makeDescriptors(const Device &dev,
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

    VkWriteDescriptorSetAccelerationStructureKHR as_update;
    DescHelper::accelStructs(desc_updates[1],
                             as_update,
                             rt_set, 
                             &tlas_data.tlas,
                             1,
                             1);

    VkDescriptorBufferInfo obj_data_info;
    obj_data_info.buffer = asset_mgr.geoAddrsBuffer.buf.buffer;
    obj_data_info.offset = 0;
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

static ImplInit setupImplInit(const BatchRenderer::Config &cfg)
{
    bool validate = enableValidation();
    bool present = debugPresent();

    void (*get_instance)() = nullptr;

    auto present_window = Optional<Window>::none();
    if (present) {
        get_instance = PresentationState::init();
        present_window =
            PresentationState::makeWindow(cfg.renderWidth, cfg.renderHeight);
    }

    Backend backend(get_instance, validate,
          !present_window.has_value(), present_window.has_value() ? 
              PresentationState::getInstanceExtensions(*present_window) :
              Span<const char *const>(nullptr, 0));

    auto present_surface = Optional<VkSurfaceKHR>::none();
    if (present) {
        present_surface =
            PresentationState::makeSurface(backend, *present_window);
    }

    return {
        validate,
        std::move(backend),
        present_window,
        present_surface,
    };
}

BatchRenderer::Impl::Impl(const Config &cfg)
    : Impl(cfg, setupImplInit(cfg))
{}

BatchRenderer::Impl::Impl(const Config &cfg, ImplInit &&init)
    : backend(std::move(init.backend)),
      dev(backend.initDevice(getVkUUIDFromCudaID(cfg.gpuID),
                             init.presentSurface)),
      mem(dev, backend),
      presentState(init.presentWindow.has_value() ?
          Optional<PresentationState>::make(backend, dev,
                            std::move(*init.presentWindow),
                            std::move(*init.presentSurface),
                            dev.gfxQF, 1, true) :
          Optional<PresentationState>::none()),
      rgbPresentIntermediate(presentState.has_value() ?
          mem.makeConversionImage(cfg.renderWidth, cfg.renderHeight,
                                  VK_FORMAT_R8G8B8A8_SRGB) :
          Optional<LocalImage>::none()),
      renderQueue(makeQueue(dev, dev.gfxQF, 0)),
      renderFence(makeFence(dev, false)),
      renderCmdPool(makeCmdPool(dev, dev.gfxQF)),
      renderCmd(makeCmdBuffer(dev, renderCmdPool)),
      shaderState(makeShaderState(dev, cfg, init)),
      pipelineState(makePipelineState(dev, shaderState)),
      fb(makeFramebuffer(dev, mem, cfg)),
      assetMgr(dev, mem, cfg.inputMode == InputMode::CPU ? -1 : cfg.gpuID,
               cfg.maxObjects),
      viewDataBuffer(cfg.inputMode == InputMode::CUDA ?
          EngineToRendererBuffer(CudaMode {}, dev, mem,
              sizeof(shader::ViewData) * fb.numViews, cfg.gpuID) :
          EngineToRendererBuffer(CpuMode {}, mem,
              sizeof(shader::ViewData) * fb.numViews)),
      viewDataAddrsBuffer(cfg.inputMode == InputMode::CUDA ?
          HostToEngineBuffer(CudaMode {}, dev, mem,
              sizeof(shader::ViewData *) * cfg.numWorlds, cfg.gpuID) :
          HostToEngineBuffer(CpuMode {}, 
              sizeof(shader::ViewData *) * cfg.numWorlds)),
      tlases(TLASData::setup(dev, GPURunUtil {
              renderCmdPool,
              renderCmd,  
              renderQueue,
              renderFence, 
          }, cfg.inputMode == InputMode::CPU ? -1 : cfg.gpuID,
          mem, cfg.numWorlds, cfg.maxInstancesPerWorld)),
      descriptors(makeDescriptors(dev, shaderState, fb, assetMgr,
                                  viewDataBuffer.rendererBuffer(), tlases)),
      launchWidth(utils::divideRoundUp(fb.renderWidth,
                                       VulkanConfig::localWorkgroupX)),
      launchHeight(utils::divideRoundUp(fb.renderHeight,
                                        VulkanConfig::localWorkgroupY)),
      renderFinished(presentState.has_value() ? makeBinarySemaphore(dev) :
                     VK_NULL_HANDLE),
      swapchainReady(presentState.has_value() ? makeBinarySemaphore(dev) :
                     VK_NULL_HANDLE),
      loadedAssets(0)
{
    auto view_data_addrs_staging_ptr =
        (shader::ViewData **)viewDataAddrsBuffer.hostPointer();

    for (CountT i = 0; i < (CountT)cfg.numWorlds; i++) {
        view_data_addrs_staging_ptr[i] = &((shader::ViewData *)
            viewDataBuffer.enginePointer())[i * cfg.maxViewsPerWorld];
    }

    if (viewDataAddrsBuffer.needsEngineCopy()) {
        GPURunUtil gpu_run {
            renderCmdPool,
            renderCmd,  
            renderQueue,
            renderFence, 
        };
        
        gpu_run.begin(dev);

        viewDataAddrsBuffer.toEngine(dev, gpu_run.cmd,
            0, sizeof(shader::ViewData *) * cfg.numWorlds);

        gpu_run.submit(dev);
    }
}

BatchRenderer::Impl::~Impl()
{
    REQ_VK(dev.dt.deviceWaitIdle(dev.hdl));

    dev.dt.destroySemaphore(dev.hdl, swapchainReady, nullptr);
    dev.dt.destroySemaphore(dev.hdl, renderFinished, nullptr);
    tlases.destroy(dev);

    dev.dt.destroyPipeline(dev.hdl, pipelineState.rt, nullptr);
    dev.dt.destroyPipelineLayout(dev.hdl, pipelineState.rtLayout, nullptr);
    dev.dt.destroyPipelineCache(dev.hdl, pipelineState.cache, nullptr);
    dev.dt.destroyCommandPool(dev.hdl, renderCmdPool, nullptr);
    dev.dt.destroyFence(dev.hdl, renderFence, nullptr);
}

CountT BatchRenderer::Impl::loadObjects(Span<const imp::SourceObject> objs)
{
    auto metadata = *assetMgr.prepareMetadata(objs);
    HostBuffer staging = mem.makeStagingBuffer(metadata.numGPUDataBytes);
    assetMgr.packAssets(staging.ptr, metadata, objs);

    Assets loaded = assetMgr.load(dev, mem, GPURunUtil {
        renderCmdPool,
        renderCmd,
        renderQueue,
        renderFence,
    }, metadata, std::move(staging));

    CountT offset = loaded.objectOffset;
    loadedAssets.emplace_back(std::move(loaded));

    return offset;
}

void BatchRenderer::Impl::render()
{
    HostEventLogging(HostEvent::renderStart);
    uint32_t swapchain_idx = 0;
    if (presentState.has_value()) {
        presentState->processInputs();

        swapchain_idx = presentState->acquireNext(dev, swapchainReady);
    }

    REQ_VK(dev.dt.resetCommandPool(dev.hdl, renderCmdPool, 0));

    VkCommandBufferBeginInfo begin_info {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    REQ_VK(dev.dt.beginCommandBuffer(renderCmd, &begin_info));

    dev.dt.cmdBindPipeline(renderCmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipelineState.rt);
    dev.dt.cmdBindDescriptorSets(renderCmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 pipelineState.rtLayout, 0, 1,
                                 &descriptors.rtSet, 0, nullptr);

    viewDataBuffer.toRenderer(dev, renderCmd,
                              VK_ACCESS_SHADER_READ_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    tlases.instanceStorage.toRenderer(dev, renderCmd,
        VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR);

    tlases.build(dev, renderCmd);

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
        launchWidth,
        launchHeight,
        fb.numViews);

    if (tlases.cudaMode) {
        dev.dt.cmdFillBuffer(
            renderCmd, tlases.devInstanceCount->buf.buffer,
            0, sizeof(uint32_t), 0);
    }

    if (presentState.has_value()) {
        VkBufferImageCopy cpy_to_present {
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            .imageOffset = { 0, 0, 0 },
            .imageExtent = { fb.renderWidth, fb.renderHeight, 1 },
        };

        VkImage present_img = presentState->getImage(swapchain_idx);

        VkBufferMemoryBarrier buffer_barrier;
        buffer_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        buffer_barrier.pNext = nullptr;
        buffer_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        buffer_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        buffer_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        buffer_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        buffer_barrier.buffer = fb.rgb.buf.buffer;
        buffer_barrier.offset = 0;
        buffer_barrier.size = VK_WHOLE_SIZE;

        std::array<VkImageMemoryBarrier, 2> layout_updates;
        layout_updates[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        layout_updates[0].pNext = nullptr;
        layout_updates[0].srcAccessMask = 0;
        layout_updates[0].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        layout_updates[0].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        layout_updates[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        layout_updates[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        layout_updates[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        layout_updates[0].image = rgbPresentIntermediate->image;
        layout_updates[0].subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        };

        dev.dt.cmdPipelineBarrier(renderCmd,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  0, 0, nullptr, 1, &buffer_barrier,
                                  1, &layout_updates[0]);

        dev.dt.cmdCopyBufferToImage(renderCmd, fb.rgb.buf.buffer,
                                    rgbPresentIntermediate->image,
                                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    1, &cpy_to_present);

        layout_updates[0].image = present_img;

        layout_updates[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        layout_updates[1].pNext = nullptr;
        layout_updates[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        layout_updates[1].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        layout_updates[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        layout_updates[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        layout_updates[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        layout_updates[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        layout_updates[1].image = rgbPresentIntermediate->image;
        layout_updates[1].subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        };

        dev.dt.cmdPipelineBarrier(renderCmd,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  0, 0, nullptr, 0, nullptr,
                                  2, &layout_updates[0]);

        VkImageBlit blit_info {
            .srcSubresource = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            .srcOffsets = {
                { 0, 0, 0 },
                { int32_t(fb.renderWidth), int32_t(fb.renderHeight), 1 },
            },
            .dstSubresource = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            .dstOffsets = {
                { 0, 0, 0 },
                { int32_t(fb.renderWidth), int32_t(fb.renderHeight), 1 },
            },
        };

        dev.dt.cmdBlitImage(renderCmd,
                            rgbPresentIntermediate->image,
                            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            present_img, 
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            1, &blit_info, VK_FILTER_LINEAR);


        layout_updates[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        layout_updates[0].dstAccessMask = 0;
        layout_updates[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        layout_updates[0].newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        dev.dt.cmdPipelineBarrier(renderCmd,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                                  0, 0, nullptr, 0, nullptr,
                                  1, &layout_updates[0]);
    }

    REQ_VK(dev.dt.endCommandBuffer(renderCmd));

    VkSubmitInfo submit_info {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.waitSemaphoreCount = 0;
    submit_info.pWaitSemaphores = nullptr;
    submit_info.pWaitDstStageMask = nullptr;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &renderCmd;

    VkPipelineStageFlags present_wait_mask =
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    if (presentState.has_value()) {
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
    HostEventLogging(HostEvent::renderEnd);

    if (!tlases.cudaMode) {
        tlases.hostInstanceCount->primitiveCount = 0;
    }
}

BatchRenderer::BatchRenderer(const Config &cfg)
    : impl_(new Impl(cfg))
{}

BatchRenderer::BatchRenderer(BatchRenderer &&o)
    : impl_(std::move(o.impl_))
{}

BatchRenderer::~BatchRenderer() = default;

CountT BatchRenderer::loadObjects(Span<const imp::SourceObject> objs)
{
    return impl_->loadObjects(objs);
}

RendererInterface BatchRenderer::getInterface() const
{
    RendererInterface renderer_iface;

    renderer_iface.tlasInstancesBase =
        (AccelStructInstance *)impl_->tlases.instanceStorage.enginePointer();

    if (impl_->tlases.cudaMode) {
        renderer_iface.numInstances =
            (AccelStructRangeInfo *)impl_->tlases.devInstanceCountCUDA->
                getDevicePointer();
    } else {
        renderer_iface.numInstances =
            impl_->tlases.hostInstanceCount;
    }

    renderer_iface.blases = 
        (uint64_t *)impl_->assetMgr.blasAddrsBuffer.enginePointer();
    renderer_iface.packedViews =  (PackedViewData **)
        impl_->viewDataAddrsBuffer.enginePointer();
    renderer_iface.numInstancesReadback =
        impl_->tlases.countReadback;

    renderer_iface.renderWidth = impl_->fb.renderWidth;
    renderer_iface.renderHeight = impl_->fb.renderHeight;

    return renderer_iface;
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

void BatchRenderer::render()
{
    impl_->render();
}

}
}
