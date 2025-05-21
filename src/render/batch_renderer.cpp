#include <stdio.h>
#include <iostream>
#include <fstream>
#include "batch_renderer.hpp"
#include "shader.hpp"
#include "ecs_interop.hpp"
#include "render_ctx.hpp"

#include <madrona/utils.hpp>
#include <madrona/heap_array.hpp>

#include <array>
#include <vector>
#include <filesystem>

#include "vk/descriptors.hpp"
#include "vk/memory.hpp"
#include "vk/utils.hpp"
#include "vk/utils.inl"

#include <chrono>

namespace madrona::render {

enum class LatestOperation {
    None,
    RenderPrepare,
    RenderViews,
    Transition,
};

namespace consts {


inline constexpr uint32_t maxDrawsPerView = 512*4;


inline constexpr uint32_t maxTextureDim = 16384;
inline constexpr uint32_t maxNumImagesX = 16;
inline constexpr uint32_t maxNumImagesY = 16;
inline constexpr uint32_t maxNumImagesPerTarget = 
    maxNumImagesX * maxNumImagesY;


// 256, is the number of views per image we can have
inline constexpr uint32_t maxDrawsPerLayeredImage = maxDrawsPerView * maxNumImagesPerTarget;
inline constexpr VkFormat colorOnlyFormat = VK_FORMAT_R8G8B8A8_UNORM;
inline constexpr VkFormat depthOnlyFormat = VK_FORMAT_R32_SFLOAT;
inline constexpr VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;
inline constexpr VkFormat outputColorFormat = VK_FORMAT_R8G8B8A8_UNORM;
inline constexpr uint32_t numDrawCmdBuffers = 4; // Triple buffering

}

////////////////////////////////////////////////////////////////////////////////
// LAYERED OUTPUT CREATION                                                    //
////////////////////////////////////////////////////////////////////////////////
// 
////////////////////////////////////////////////////////////////////////////////
// LAYERED OUTPUT CREATION                                                    //
////////////////////////////////////////////////////////////////////////////////

// A layered target will have a color image with the max amount of layers, depth
// image with max amount of layers.

static HeapArray<LayeredTarget> makeLayeredTargets(uint32_t width,
                                                   uint32_t height,
                                                   uint32_t max_num_views,
                                                   const vk::Device &dev,
                                                   vk::MemoryAllocator &alloc,
                                                   bool depth_only)
{
    uint32_t max_image_dim_x = std::min(consts::maxTextureDim, consts::maxNumImagesX * width);
    uint32_t max_image_dim_y = std::min(consts::maxTextureDim, consts::maxNumImagesY * height);
    // Each view is going to be stored in one section of the layer (one viewport of
    // the layer). Each image, will have as many layers as possible.
    uint32_t max_images_x = max_image_dim_x / width;
    uint32_t max_images_y = max_image_dim_y / height;

    uint32_t max_views_per_target = max_images_x * max_images_y;

    // Number of images to allocate
    uint32_t num_targets = utils::divideRoundUp(max_num_views,
                                                max_views_per_target);

    HeapArray<LayeredTarget> local_images (num_targets);

    // Total number of layers
    uint32_t views_left = max_num_views;

    for (int i = 0; i < (int)num_targets; ++i) {
        uint32_t num_views_in_image = std::min((uint32_t)views_left,
                                               max_views_per_target);

        uint32_t image_width = width * std::min(num_views_in_image, max_images_x);

        uint32_t image_height = height * 
            utils::divideRoundUp(num_views_in_image, max_images_x);

        LayeredTarget target = {
            .vizBuffer = alloc.makeColorAttachment(image_width, image_height,
                                                   1,
                                                   depth_only ? consts::depthOnlyFormat : consts::colorOnlyFormat),
            .vizBufferView = {},
            .depth = alloc.makeDepthAttachment(image_width, image_height,
                                               1,
                                               consts::depthFormat),
            .depthView = {},
            .numViews = num_views_in_image,
            .lightingSet = {},
            .pixelWidth = image_width,
            .pixelHeight = image_height,
            .viewWidth = width,
            .viewHeight = height,
        };

        VkImageViewCreateInfo view_info = {};

        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        view_info.subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
        };

        view_info.image = target.vizBuffer.image;
        view_info.format = depth_only ? consts::depthOnlyFormat : consts::colorOnlyFormat;
        REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &target.vizBufferView));

        view_info.image = target.depth.image;
        view_info.format = consts::depthFormat;
        view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &target.depthView));

        local_images.emplace(i, std::move(target));

        views_left -= num_views_in_image;
    }

    return local_images;
}

////////////////////////////////////////////////////////////////////////////////
// DRAW COMMAND BUFFER CREATION                                               //
////////////////////////////////////////////////////////////////////////////////
struct DrawCommandPackage {
    // Draw cmds and drawdata
    vk::LocalBuffer drawBuffer;

    // This descriptor set contains draw information
    VkDescriptorSet drawBufferSetPrepare;
    VkDescriptorSet drawBufferSetDraw;

    uint32_t drawCmdOffset;
    uint32_t drawCmdBufferSize;

    uint32_t numDrawCounts;
};


////////////////////////////////////////////////////////////////////////////////
// RENDER PIPELINE CREATION                                                   //
////////////////////////////////////////////////////////////////////////////////
static vk::PipelineShaders makeDrawShaders(const vk::Device &dev, 
                                           VkSampler repeat_sampler,
                                           VkSampler clamp_sampler,
                                           bool depth_only)
{
    (void)repeat_sampler;
    (void)clamp_sampler;

    std::filesystem::path shader_dir =
        std::filesystem::path(STRINGIFY(MADRONA_RENDER_DATA_DIR)) /
        "shaders";

    auto shader_path = [depth_only, shader_dir] () {
        if (depth_only) {
            return (shader_dir / "batch_draw_depth.hlsl").string();
        } else {
            return (shader_dir / "batch_draw_rgb.hlsl").string();
        }
    } ();

    ShaderCompiler compiler;
    SPIRVShader vert_spirv = compiler.compileHLSLFileToSPV(
        shader_path.c_str(), {}, {},
        { "vert", ShaderStage::Vertex });

    SPIRVShader frag_spirv = compiler.compileHLSLFileToSPV(
        shader_path.c_str(), {}, {},
        { "frag", ShaderStage::Fragment });

    std::array<SPIRVShader, 2> shaders {
        std::move(vert_spirv),
        std::move(frag_spirv),
    };

    if (depth_only) {
        StackAlloc tmp_alloc;
        return vk::PipelineShaders(dev, tmp_alloc, shaders,
            Span<const vk::BindingOverride>({}));
    } else {
        StackAlloc tmp_alloc;
        return vk::PipelineShaders(dev, tmp_alloc, shaders,
            Span<const vk::BindingOverride>({
                 vk::BindingOverride {
                     3, 0, VK_NULL_HANDLE,
                     InternalConfig::maxTextures, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT
                 },
                 vk::BindingOverride {
                     3, 1, repeat_sampler, 1, 0
                 }
            }));
    }
}

static PipelineMP<1> makeDrawPipeline(const vk::Device &dev,
                                    VkPipelineCache pipeline_cache,
                                    VkRenderPass render_pass,
                                    uint32_t num_frames,
                                    uint32_t num_pools,
                                    bool depth_only,
                                    VkSampler repeat_sampler)
{
    auto shaders = makeDrawShaders(dev, repeat_sampler, repeat_sampler, depth_only);

    VkPipelineVertexInputStateCreateInfo vert_info {};
    VkPipelineInputAssemblyStateCreateInfo input_assembly_info {};
    VkPipelineViewportStateCreateInfo viewport_info {};
    VkPipelineMultisampleStateCreateInfo multisample_info {};
    VkPipelineRasterizationStateCreateInfo raster_info {};

    initCommonDrawPipelineInfo(vert_info, input_assembly_info, 
        viewport_info, multisample_info, raster_info);

    // Depth/Stencil
    VkPipelineDepthStencilStateCreateInfo depth_info {};
    depth_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth_info.depthTestEnable = VK_TRUE;
    depth_info.depthWriteEnable = VK_TRUE;
    depth_info.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;
    depth_info.depthBoundsTestEnable = VK_FALSE;
    depth_info.stencilTestEnable = VK_FALSE;
    depth_info.back.compareOp = VK_COMPARE_OP_ALWAYS;

    // Blend
    VkPipelineColorBlendAttachmentState blend_attach {};
    blend_attach.blendEnable = VK_FALSE;

    if (depth_only) {
        blend_attach.colorWriteMask = VK_COLOR_COMPONENT_R_BIT;
    } else {
        blend_attach.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                                      VK_COLOR_COMPONENT_G_BIT |
                                      VK_COLOR_COMPONENT_B_BIT |
                                      VK_COLOR_COMPONENT_A_BIT;
    }

    std::array<VkPipelineColorBlendAttachmentState, 1> blend_attachments {{
        blend_attach
    }};

    VkPipelineColorBlendStateCreateInfo blend_info {};
    blend_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend_info.logicOpEnable = VK_FALSE;
    blend_info.attachmentCount = 
        static_cast<uint32_t>(blend_attachments.size());
    blend_info.pAttachments = blend_attachments.data();

    // Dynamic
    std::array dyn_enable {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };

    VkPipelineDynamicStateCreateInfo dyn_info {};
    dyn_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn_info.dynamicStateCount = dyn_enable.size();
    dyn_info.pDynamicStates = dyn_enable.data();

    VkPushConstantRange push_const {
        VK_SHADER_STAGE_VERTEX_BIT,
        0,
        sizeof(shader::BatchDrawPushConst),
    };

    // Layout configuration

    uint32_t num_layouts = depth_only ? 3 : 4;
    std::array<VkDescriptorSetLayout, 4> draw_desc_layouts {{
        shaders.getLayout(0),
        shaders.getLayout(1),
        shaders.getLayout(2)
    }};

    if (!depth_only) {
        draw_desc_layouts[3] = shaders.getLayout(3);
    }

    VkPipelineLayoutCreateInfo gfx_layout_info;
    gfx_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    gfx_layout_info.pNext = nullptr;
    gfx_layout_info.flags = 0;
    gfx_layout_info.setLayoutCount =
        static_cast<uint32_t>(num_layouts);
    gfx_layout_info.pSetLayouts = draw_desc_layouts.data();
    gfx_layout_info.pushConstantRangeCount = 1;
    gfx_layout_info.pPushConstantRanges = &push_const;

    VkPipelineLayout draw_layout;
    REQ_VK(dev.dt.createPipelineLayout(dev.hdl, &gfx_layout_info, nullptr,
                                       &draw_layout));

    std::array<VkPipelineShaderStageCreateInfo, 2> gfx_stages {{
        {
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            nullptr,
            0,
            VK_SHADER_STAGE_VERTEX_BIT,
            shaders.getShader(0),
            "vert",
            nullptr,
        },
        {
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            nullptr,
            0,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            shaders.getShader(1),
            "frag",
            nullptr,
        },
    }};

    VkFormat color_format = depth_only ? consts::depthOnlyFormat : consts::colorOnlyFormat;
    VkFormat depth_format = consts::depthFormat;

    VkPipelineRenderingCreateInfo rendering_info = {};
    rendering_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rendering_info.colorAttachmentCount = 1;
    rendering_info.pColorAttachmentFormats = &color_format;
    rendering_info.depthAttachmentFormat = depth_format;

    VkGraphicsPipelineCreateInfo gfx_info;
    gfx_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gfx_info.pNext = &rendering_info;
    gfx_info.flags = 0;
    gfx_info.stageCount = gfx_stages.size();
    gfx_info.pStages = gfx_stages.data();
    gfx_info.pVertexInputState = &vert_info;
    gfx_info.pInputAssemblyState = &input_assembly_info;
    gfx_info.pTessellationState = nullptr;
    gfx_info.pViewportState = &viewport_info;
    gfx_info.pRasterizationState = &raster_info;
    gfx_info.pMultisampleState = &multisample_info;
    gfx_info.pDepthStencilState = &depth_info;
    gfx_info.pColorBlendState = &blend_info;
    gfx_info.pDynamicState = &dyn_info;
    gfx_info.layout = draw_layout;
    gfx_info.renderPass = render_pass;
    gfx_info.subpass = 0;
    gfx_info.basePipelineHandle = VK_NULL_HANDLE;
    gfx_info.basePipelineIndex = -1;

    VkPipeline draw_pipeline;
    REQ_VK(dev.dt.createGraphicsPipelines(dev.hdl, pipeline_cache, 1,
                                          &gfx_info, nullptr, &draw_pipeline));

    // std::array<vk::FixedDescriptorPool, D> desc_pools;
    DynArray<vk::FixedDescriptorPool> desc_pools(num_pools);
    for (int i = 0; i < (int)num_pools; ++i) {
        desc_pools.emplace_back(dev, shaders, i, num_frames);
    }

    return {
        std::move(shaders),
        draw_layout,
        { draw_pipeline },
        std::move(desc_pools)
    };
}

////////////////////////////////////////////////////////////////////////////////
// GENERIC COMPUTE PIPELINE CREATION                                          //
////////////////////////////////////////////////////////////////////////////////

static vk::PipelineShaders makeShaders(const vk::Device &dev,
                                       const char *shader_file,
                                       const char *func_name = "main",
                                       VkSampler sampler = VK_NULL_HANDLE)
{
    (void)sampler;

    std::filesystem::path shader_dir =
        std::filesystem::path(STRINGIFY(MADRONA_RENDER_DATA_DIR)) /
        "shaders";

    ShaderCompiler compiler;
    SPIRVShader spirv = compiler.compileHLSLFileToSPV(
        (shader_dir / shader_file).string().c_str(), {},
        {}, {func_name, ShaderStage::Compute });
    
    StackAlloc tmp_alloc;
    return vk::PipelineShaders(dev, tmp_alloc,
                               Span<const SPIRVShader>(&spirv, 1), {});
}

static vk::PipelineShaders makeShadersLighting(const vk::Device &dev,
                                       const char *shader_file,
                                       const char *func_name = "main",
                                       VkSampler repeat_sampler = VK_NULL_HANDLE)
{
    std::filesystem::path shader_dir =
        std::filesystem::path(STRINGIFY(MADRONA_RENDER_DATA_DIR)) /
        "shaders";

    ShaderCompiler compiler;
    SPIRVShader spirv = compiler.compileHLSLFileToSPV(
        (shader_dir / shader_file).string().c_str(), {},
        {}, {func_name, ShaderStage::Compute });
    
    StackAlloc tmp_alloc;
    return vk::PipelineShaders(dev, tmp_alloc,
                               Span<const SPIRVShader>(&spirv, 1), 
                               Span<const vk::BindingOverride>({
                                   vk::BindingOverride {
                                       0, 0, VK_NULL_HANDLE, 
                                       100, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT 
                                   },
                                   vk::BindingOverride {
                                       0, 3, VK_NULL_HANDLE,
                                       100, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT
                                   },
                                   vk::BindingOverride {
                                       0, 4, repeat_sampler, 1, 0
                                   },
                                }));
}

template <typename T>
static PipelineMP<1> makeComputePipeline(const vk::Device &dev,
                                            VkPipelineCache pipeline_cache,
                                            uint32_t num_pools,
                                            uint32_t push_constant_size,
                                            uint32_t num_descriptor_sets,
                                            VkSampler repeat_sampler,
                                            const char *shader_file,
                                            bool depth_only = false,
                                            const char *func_name = "main",
                                            T make_shaders_proc = makeShaders)
{
    (void)depth_only;

    vk::PipelineShaders shader = make_shaders_proc(dev, shader_file, func_name, repeat_sampler);

    VkPushConstantRange push_const = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = push_constant_size
    };

    std::vector<VkDescriptorSetLayout> desc_layouts(shader.getLayoutCount());
    for (uint32_t i = 0; i < shader.getLayoutCount(); ++i) {
        desc_layouts[i] = shader.getLayout(i);
    }

    VkPipelineLayoutCreateInfo layout_info;
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.pNext = nullptr;
    layout_info.flags = 0;
    layout_info.setLayoutCount = static_cast<uint32_t>(desc_layouts.size());
    layout_info.pSetLayouts = desc_layouts.data();
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push_const;

    VkPipelineLayout layout;
    REQ_VK(dev.dt.createPipelineLayout(dev.hdl, &layout_info, nullptr,
                                       &layout));

    std::array<VkComputePipelineCreateInfo, 1> compute_infos;

    compute_infos[0].sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compute_infos[0].pNext = nullptr;
    compute_infos[0].flags = 0;
    compute_infos[0].stage = {
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        nullptr, //&subgroup_size,
        VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT,
        VK_SHADER_STAGE_COMPUTE_BIT,
        shader.getShader(0),
        func_name,
        nullptr,
    };
    compute_infos[0].layout = layout;
    compute_infos[0].basePipelineHandle = VK_NULL_HANDLE;
    compute_infos[0].basePipelineIndex = -1;

    std::array<VkPipeline, compute_infos.size()> pipelines;
    REQ_VK(dev.dt.createComputePipelines(dev.hdl, pipeline_cache,
                                         compute_infos.size(),
                                         compute_infos.data(), nullptr,
                                         pipelines.data()));

    DynArray<vk::FixedDescriptorPool> desc_pools(num_pools);
    for (int i = 0; i < (int)num_pools; ++i) {
        desc_pools.emplace_back(dev, shader, i, num_descriptor_sets);
    }

    return PipelineMP<1> {
        std::move(shader),
        layout,
        pipelines,
        std::move(desc_pools)
    };
}

struct BatchFrame {
    BatchImportedBuffers buffers;

    vk::LocalBuffer lighting;
    vk::HostBuffer lightingStaging;

    vk::LocalBuffer skyInput;
    vk::HostBuffer skyInputStaging;

    // View, instance info, instance data
    VkDescriptorSet viewInstanceSetPrepare;
    VkDescriptorSet viewAABBSetPrepare;
    VkDescriptorSet viewInstanceSetDraw;
    VkDescriptorSet viewInstanceSetLighting;

    HeapArray<LayeredTarget> targets;
    vk::DedicatedBuffer rgbOutput;
    vk::DedicatedBuffer depthOutput;

#ifdef MADRONA_VK_CUDA_SUPPORT
    vk::CudaImportedBuffer rgbOutputCUDA;
    vk::CudaImportedBuffer depthOutputCUDA;
#endif

    // Swapchain of draw packages which get used to feed to the rasterizer
    HeapArray<DrawCommandPackage> drawPackageSwapchain;

    // Descriptor set which contains all the vizBuffer outputs and
    // the lighting outputs
    VkDescriptorSet targetsSetLighting;
    VkDescriptorSet pbrSet;

    VkCommandPool prepareCmdPool;
    VkCommandBuffer prepareCmdbuf;

    VkCommandPool renderCmdPool;
    VkCommandBuffer renderCmdbuf;

    // Waited for by the viewer or the batch renderer
    VkSemaphore prepareFinished;

    // Waited for by the viewer to render stuff to the window
    VkSemaphore renderFinished;

    // Waited for if that latest thing was a transition
    VkSemaphore layoutTransitionFinished;

    // Waited for at the beginning of each renderViews call
    VkFence prepareFence;
    VkFence renderFence;

    // Keep track of which semaphore to wait on
    LatestOperation latestOp;

    VkFence & getLatestFence()
    {
        if (latestOp == LatestOperation::RenderPrepare) {
            return prepareFence;
        } else {
            return renderFence;
        }
    }
};

static DrawCommandPackage makeDrawCommandPackage(vk::Device& dev,
                          render::vk::MemoryAllocator &alloc,
                          PipelineMP<1> &prepare_views,
                          PipelineMP<1> &draw_views,
                          uint32_t max_views_per_target)
{
    VkDescriptorSet prepare_set = prepare_views.descPools[1].makeSet();
    VkDescriptorSet draw_set = draw_views.descPools[1].makeSet();

    // Make Draw Buffers
    int64_t buffer_offsets[2];
    int64_t buffer_sizes[3] = {
        (int64_t)sizeof(uint32_t) * max_views_per_target,
        (int64_t)sizeof(shader::DrawCmd) * consts::maxDrawsPerLayeredImage,
        (int64_t)sizeof(shader::DrawDataBR) * consts::maxDrawsPerLayeredImage
    };

    int64_t num_draw_bytes = utils::computeBufferOffsets(
        buffer_sizes, buffer_offsets, 256);

    vk::LocalBuffer drawBuffer = alloc.makeLocalBuffer(num_draw_bytes).value();

    std::array<VkWriteDescriptorSet, 6> desc_updates;

    VkDescriptorBufferInfo draw_count_info;
    draw_count_info.buffer = drawBuffer.buffer;
    draw_count_info.offset = 0;
    draw_count_info.range = buffer_sizes[0];

    vk::DescHelper::storage(desc_updates[0], prepare_set, &draw_count_info, 0);
    vk::DescHelper::storage(desc_updates[3], draw_set, &draw_count_info, 0);

    VkDescriptorBufferInfo draw_cmd_info;
    draw_cmd_info.buffer = drawBuffer.buffer;
    draw_cmd_info.offset = buffer_offsets[0];
    draw_cmd_info.range = buffer_sizes[1];

    vk::DescHelper::storage(desc_updates[1], prepare_set, &draw_cmd_info, 1);
    vk::DescHelper::storage(desc_updates[4], draw_set, &draw_cmd_info, 1);

    VkDescriptorBufferInfo draw_data_info;
    draw_data_info.buffer = drawBuffer.buffer;
    draw_data_info.offset = buffer_offsets[1];
    draw_data_info.range = buffer_sizes[2];

    vk::DescHelper::storage(desc_updates[2], prepare_set, &draw_data_info, 2);
    vk::DescHelper::storage(desc_updates[5], draw_set, &draw_data_info, 2);

    vk::DescHelper::update(dev, desc_updates.data(), desc_updates.size());

    return DrawCommandPackage {
        std::move(drawBuffer),
        prepare_set,
        draw_set,
        (uint32_t)buffer_offsets[0],
        (uint32_t)num_draw_bytes,
        max_views_per_target
    };
}

static void makeBatchFrame(vk::Device& dev, 
                           BatchFrame* frame,
                           render::vk::MemoryAllocator &alloc,
                           const BatchRenderer::Config &cfg,
                           PipelineMP<1> &prepare_views,
                           Optional<PipelineMP<1>> &draw,
                           VkDescriptorSet lighting_set,
                           VkDescriptorSet pbr_set,
                           bool enable_batch_renderer,
                           RenderContext &rctx,
                           uint32_t view_width,
                           uint32_t view_height,
                           bool depth_only)
{
    VkDeviceSize lights_size = InternalConfig::maxLights * sizeof(render::shader::LightDesc);
    vk::LocalBuffer lights = alloc.makeLocalBuffer(lights_size).value();
    vk::HostBuffer lights_staging = alloc.makeStagingBuffer(lights_size);

    VkDeviceSize sky_input_size = sizeof(render::shader::SkyData);
    vk::LocalBuffer sky_input = alloc.makeLocalBuffer(sky_input_size).value();
    vk::HostBuffer sky_input_staging = alloc.makeStagingBuffer(sky_input_size);

    VkDeviceSize view_size = (cfg.numWorlds * cfg.maxViewsPerWorld) * sizeof(PerspectiveCameraData);
    vk::LocalBuffer views = alloc.makeLocalBuffer(view_size).value();

    VkDeviceSize instance_size = (cfg.numWorlds * cfg.maxInstancesPerWorld) * sizeof(InstanceData);
    vk::LocalBuffer instances = alloc.makeLocalBuffer(instance_size).value();

    VkDeviceSize instance_offset_size = (cfg.numWorlds) * sizeof(uint32_t);
    vk::LocalBuffer instance_offsets = alloc.makeLocalBuffer(instance_offset_size).value();

    VkDeviceSize view_offset_size = (cfg.numWorlds) * sizeof(uint32_t);
    vk::LocalBuffer view_offsets = alloc.makeLocalBuffer(view_offset_size).value();

    VkDeviceSize light_offsets_size = cfg.numWorlds * sizeof(uint32_t);
    vk::LocalBuffer light_offsets = alloc.makeLocalBuffer(light_offsets_size).value();

    VkCommandPool prepare_cmdpool = vk::makeCmdPool(dev, dev.gfxQF);
    VkCommandPool render_cmdpool = vk::makeCmdPool(dev, dev.gfxQF);
    VkCommandBuffer prepare_cmdbuf = vk::makeCmdBuffer(dev, prepare_cmdpool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
    VkCommandBuffer render_cmdbuf = vk::makeCmdBuffer(dev, render_cmdpool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);

    VkSemaphore prepare_finished = vk::makeBinarySemaphore(dev);
    VkSemaphore render_finished = vk::makeBinarySemaphore(dev);
    VkSemaphore transition_finished = vk::makeBinarySemaphore(dev);

    VkFence prepare_fence = vk::makeFence(dev, true);
    VkFence render_fence = vk::makeFence(dev, true);

    const bool supports_cuda_export =
#ifdef MADRONA_VK_CUDA_SUPPORT
        true;
#else
        false;
#endif

    if (!enable_batch_renderer) {
        // FIXME get rid of the need for these fake buffers
        auto fake_rgb_buf = alloc.makeDedicatedBuffer(
            1, false, supports_cuda_export);
        auto fake_depth_buf = alloc.makeDedicatedBuffer(
            1, false, supports_cuda_export);

#ifdef MADRONA_VK_CUDA_SUPPORT
        vk::CudaImportedBuffer fake_rgb_cuda(dev, fake_rgb_buf.mem, 1);
        vk::CudaImportedBuffer fake_depth_cuda(dev, fake_depth_buf.mem, 1);
#endif

        new (frame) BatchFrame{
            { std::move(views), std::move(view_offsets), std::move(instances), std::move(instance_offsets), std::move(lights), std::move(light_offsets) },
            std::move(lights), std::move(lights_staging),
            std::move(sky_input), std::move(sky_input_staging),
            VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE,
            HeapArray<LayeredTarget>(0),
            std::move(fake_rgb_buf),
            std::move(fake_depth_buf),
#ifdef MADRONA_VK_CUDA_SUPPORT
            std::move(fake_rgb_cuda),
            std::move(fake_depth_cuda),
#endif
            HeapArray<DrawCommandPackage>(0),
            VK_NULL_HANDLE, VK_NULL_HANDLE,
            prepare_cmdpool,
            prepare_cmdbuf,
            render_cmdpool,
            render_cmdbuf,
            prepare_finished,
            render_finished,
            transition_finished,
            prepare_fence,
            render_fence,
            LatestOperation::None
        };

        return;
    }

    VkDescriptorSet prepare_views_set = prepare_views.descPools[0].makeSet();
    VkDescriptorSet draw_views_set = draw->descPools[0].makeSet();
    VkDescriptorSet aabb_set = prepare_views.descPools[3].makeSet();

    //Descriptor sets
    std::array<VkWriteDescriptorSet, 11> desc_updates;

    VkDescriptorBufferInfo view_info;
    view_info.buffer = views.buffer;
    view_info.offset = 0;
    view_info.range = view_size;
    vk::DescHelper::storage(desc_updates[0], prepare_views_set, &view_info, 0);
    vk::DescHelper::storage(desc_updates[1], draw_views_set, &view_info, 0);

    VkDescriptorBufferInfo instance_info;
    instance_info.buffer = instances.buffer;
    instance_info.offset = 0;
    instance_info.range = instance_size;
    vk::DescHelper::storage(desc_updates[2], prepare_views_set, &instance_info, 1);
    vk::DescHelper::storage(desc_updates[3], draw_views_set, &instance_info, 1);

    VkDescriptorBufferInfo offset_info;
    offset_info.buffer = instance_offsets.buffer;
    offset_info.offset = 0;
    offset_info.range = instance_offset_size;
    vk::DescHelper::storage(desc_updates[4], prepare_views_set, &offset_info, 2);
    vk::DescHelper::storage(desc_updates[5], draw_views_set, &offset_info, 2);

#if 0
    VkDescriptorBufferInfo aabb_info;
    aabb_info.buffer = aabbs.buffer;
    aabb_info.offset = 0;
    aabb_info.range = aabb_size;
    vk::DescHelper::storage(desc_updates[6], aabb_set, &aabb_info, 0);
#endif

    // PBR descriptor sets

    VkDescriptorBufferInfo light_data_info;
    light_data_info.buffer = lights.buffer;
    light_data_info.offset = 0;
    light_data_info.range = VK_WHOLE_SIZE;

    vk::DescHelper::storage(desc_updates[6],
                            pbr_set, &light_data_info, 0);
    vk::DescHelper::storage(desc_updates[7], draw_views_set, &light_data_info, 3);

    VkDescriptorImageInfo transmittance_info;
    transmittance_info.imageView = rctx.sky_.transmittanceView;
    transmittance_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    transmittance_info.sampler = VK_NULL_HANDLE;

    vk::DescHelper::textures(desc_updates[8],
                             pbr_set, &transmittance_info, 1, 1);

    VkDescriptorImageInfo irradiance_info;
    irradiance_info.imageView = rctx.sky_.irradianceView;
    irradiance_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    irradiance_info.sampler = VK_NULL_HANDLE;

    vk::DescHelper::textures(desc_updates[9],
                             pbr_set, &irradiance_info, 1, 2);

    VkDescriptorImageInfo scattering_info;
    scattering_info.imageView = rctx.sky_.scatteringView;
    scattering_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    scattering_info.sampler = VK_NULL_HANDLE;

    vk::DescHelper::textures(desc_updates[10],
                             pbr_set, &scattering_info, 1, 3);

    VkDescriptorBufferInfo sky_info;
    sky_info.buffer = sky_input.buffer;
    sky_info.offset = 0;
    sky_info.range = VK_WHOLE_SIZE;

    vk::DescHelper::storage(desc_updates[11],
                            pbr_set, &sky_info, 4);

    vk::DescHelper::update(dev, desc_updates.data(), desc_updates.size());

    HeapArray<DrawCommandPackage> draw_packages(consts::numDrawCmdBuffers);
    for (int i = 0; i < (int)consts::numDrawCmdBuffers; ++i) {
        uint32_t max_image_dim_x = std::min(consts::maxTextureDim, consts::maxNumImagesX * view_width);
        uint32_t max_image_dim_y = std::min(consts::maxTextureDim, consts::maxNumImagesY * view_height);

        // Each view is going to be stored in one section of the layer (one viewport of
        // the layer). Each image, will have as many layers as possible.
        uint32_t max_images_x = max_image_dim_x / view_width;
        uint32_t max_images_y = max_image_dim_y / view_height;

        uint32_t max_views_per_target = max_images_x * max_images_y;

        draw_packages.emplace(i, makeDrawCommandPackage(
                    dev, alloc, prepare_views, *draw,
                    max_views_per_target));
    }

    HeapArray<LayeredTarget> layered_targets = makeLayeredTargets(
        cfg.renderWidth, cfg.renderHeight, 
        cfg.numWorlds * cfg.maxViewsPerWorld,
        dev, alloc,
        depth_only);

    uint64_t total_num_pixels = 
        (uint64_t)cfg.renderWidth * (uint64_t)cfg.renderHeight * 
        (uint64_t)cfg.numWorlds * (uint64_t)cfg.maxViewsPerWorld;

    uint64_t num_rgb_bytes = total_num_pixels * sizeof(uint8_t) * 4_u64;
    uint64_t num_depth_bytes = total_num_pixels * sizeof(float);

    vk::DedicatedBuffer rgb_output_buffer = alloc.makeDedicatedBuffer(
        num_rgb_bytes, false, supports_cuda_export);

    vk::DedicatedBuffer depth_output_buffer = alloc.makeDedicatedBuffer(
        num_depth_bytes, false, supports_cuda_export);

#ifdef MADRONA_VK_CUDA_SUPPORT
    vk::CudaImportedBuffer rgb_output_cuda(
        dev, rgb_output_buffer.mem, num_rgb_bytes);

    vk::CudaImportedBuffer depth_output_cuda(
        dev, depth_output_buffer.mem, num_depth_bytes);
#endif

    {
        // Update lighting_set to point to the layered vbuffer and 
        // output buffer
        HeapArray<VkWriteDescriptorSet> lighting_desc_updates(
            2*layered_targets.size() + 2);
        HeapArray<VkDescriptorImageInfo> vbuffer_infos(
            layered_targets.size());
        HeapArray<VkDescriptorImageInfo> depth_buffer_infos(
            layered_targets.size());

        for (CountT i = 0; i < layered_targets.size(); ++i) {
            vbuffer_infos[i] = {
                .sampler = VK_NULL_HANDLE,
                .imageView = layered_targets[i].vizBufferView,
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
            };

            depth_buffer_infos[i] = {
                .sampler = VK_NULL_HANDLE,
                .imageView = layered_targets[i].depthView,
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
            };

            vk::DescHelper::storageImage(lighting_desc_updates[i*2],
                                         lighting_set, 
                                         &vbuffer_infos[i],
                                         0, i);

            vk::DescHelper::textures(lighting_desc_updates[i*2 + 1],
                                         lighting_set, 
                                         &depth_buffer_infos[i],
                                         1,
                                         3, i);
        }

        VkDescriptorBufferInfo rgb_buffer_info {
            .buffer = rgb_output_buffer.buf.buffer,
            .offset = 0,
            .range = VK_WHOLE_SIZE,
        };

        vk::DescHelper::storage(
            lighting_desc_updates[lighting_desc_updates.size() - 2], 
            lighting_set, 
            &rgb_buffer_info,
            1);

        VkDescriptorBufferInfo depth_buffer_info {
            .buffer = depth_output_buffer.buf.buffer,
            .offset = 0,
            .range = VK_WHOLE_SIZE,
        };

        vk::DescHelper::storage(
            lighting_desc_updates[lighting_desc_updates.size() - 1], 
            lighting_set, 
            &depth_buffer_info,
            2);

        vk::DescHelper::update(dev, lighting_desc_updates.data(),
                               lighting_desc_updates.size());
    }

    new (frame) BatchFrame{
        {
            std::move(views),
            std::move(view_offsets),
            std::move(instances),
            std::move(instance_offsets),
            std::move(lights),
            std::move(light_offsets)
        },
        std::move(lights),
        std::move(lights_staging),
        std::move(sky_input),
        std::move(sky_input_staging),
        prepare_views_set,
        aabb_set,
        draw_views_set,
        prepare_views_set,
        std::move(layered_targets),
        std::move(rgb_output_buffer),
        std::move(depth_output_buffer),
#ifdef MADRONA_VK_CUDA_SUPPORT
        std::move(rgb_output_cuda),
        std::move(depth_output_cuda),
#endif
        std::move(draw_packages),
        lighting_set,
        pbr_set,
        prepare_cmdpool,
        prepare_cmdbuf,
        render_cmdpool,
        render_cmdbuf,
        prepare_finished,
        render_finished,
        transition_finished,
        prepare_fence,
        render_fence,
        LatestOperation::None
    };
}

////////////////////////////////////////////////////////////////////////////////
// RASTERIZATION AND RENDERING / POST PROCESSING                              //
////////////////////////////////////////////////////////////////////////////////
static void issueRasterLayoutTransitions(vk::Device &dev, 
                                   LayeredTarget &target,
                                   VkCommandBuffer &draw_cmd)
{
    // Transition image layouts
    std::array barriers = {
        VkImageMemoryBarrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_NONE,
            .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = target.vizBuffer.image,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        },
        VkImageMemoryBarrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_NONE,
            .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = target.depth.image,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        },
    };

    dev.dt.cmdPipelineBarrier(draw_cmd,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
            0, 0, nullptr, 0, nullptr,
            barriers.size(), barriers.data());   
}

static void issueComputeLayoutTransitions(vk::Device &dev, 
                                   LayeredTarget &target,
                                   VkCommandBuffer &draw_cmd)
{
    // Transition image layouts
    std::array barriers = {
        VkImageMemoryBarrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = target.vizBuffer.image,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        },

        VkImageMemoryBarrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = target.depth.image,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        },
    };

    dev.dt.cmdPipelineBarrier(draw_cmd,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 0, nullptr,
        barriers.size(), barriers.data());   
}

static void issueRasterization(vk::Device &dev, 
                               PipelineMP<1> &draw_pipeline, 
                               LayeredTarget &target,
                               VkCommandBuffer &draw_cmd,
                               DrawCommandPackage &view_batch,
                               BatchFrame &batch_frame,
                               VkDescriptorSet asset_set,
                               VkDescriptorSet asset_mat_tex_set,
                               VkExtent2D render_extent,
                               const DynArray<AssetData> &loaded_assets,
                               bool depth_only)
{
    (void)render_extent;

    VkRenderingAttachmentInfoKHR color_attach = {};
    color_attach.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
    color_attach.imageView = target.vizBufferView;
    color_attach.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color_attach.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attach.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingAttachmentInfoKHR depth_attach = {};
    depth_attach.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR;
    depth_attach.imageView = target.depthView;
    depth_attach.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depth_attach.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attach.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRect2D total_rect = {
        .offset = {},
        .extent = { target.pixelWidth, target.pixelHeight }
    };

    VkRenderingInfo rendering_info = {};
    rendering_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    rendering_info.renderArea = total_rect;
    rendering_info.layerCount = 1;
    rendering_info.colorAttachmentCount = 1;
    rendering_info.pColorAttachments = &color_attach;
    rendering_info.pDepthAttachment = &depth_attach;

    dev.dt.cmdBeginRenderingKHR(draw_cmd, &rendering_info);

    dev.dt.cmdBindPipeline(draw_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                           draw_pipeline.hdls[0]);

    dev.dt.cmdBindIndexBuffer(draw_cmd, loaded_assets[0].buf.buffer,
                              loaded_assets[0].idxBufferOffset,
                              VK_INDEX_TYPE_UINT32);

    uint32_t num_descriptors = depth_only ? 3 : 4;
    std::array<VkDescriptorSet, 4> draw_descriptors = {
        batch_frame.viewInstanceSetDraw,
        view_batch.drawBufferSetDraw,
        asset_set,
        asset_mat_tex_set
    };

    dev.dt.cmdBindDescriptorSets(draw_cmd,
                                 VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 draw_pipeline.layout,
                                 0, 
                                 num_descriptors,
                                 draw_descriptors.data(), 
                                 0, nullptr);

    uint32_t max_image_dim_x = std::min(consts::maxTextureDim, consts::maxNumImagesX * target.viewWidth);
    uint32_t max_num_image_x = max_image_dim_x / target.viewWidth;

    for (uint32_t i = 0; i < target.numViews; ++i) {
        uint32_t image_x = i % max_num_image_x;
        uint32_t image_y = i / max_num_image_x;

        uint32_t count_offset = i * sizeof(uint32_t);

        // Set viewport and scissor
        VkViewport viewport = {
            .x = (float)(image_x * target.viewWidth),
            .y = (float)(image_y * target.viewHeight),
            .width = (float)target.viewWidth,
            .height = (float)target.viewHeight,
            .minDepth = 0.f,
            .maxDepth = 1.f
        };

        VkRect2D rect = {
            .offset = { (int32_t)(image_x * target.viewWidth),
                        (int32_t)(image_y * target.viewHeight) },
            .extent = { (uint32_t)target.viewWidth,
                        (uint32_t)target.viewHeight }
        };

        dev.dt.cmdSetViewport(draw_cmd, 0, 1, &viewport);
        dev.dt.cmdSetScissor(draw_cmd, 0, 1, &rect);

        shader::BatchDrawPushConst push_const = {
            i * consts::maxDrawsPerView
        };

        dev.dt.cmdPushConstants(draw_cmd, draw_pipeline.layout,
                                VK_SHADER_STAGE_VERTEX_BIT, 0,
                                sizeof(push_const),
                                &push_const);

        dev.dt.cmdDrawIndexedIndirectCount(draw_cmd, 
                                           view_batch.drawBuffer.buffer,
                                           view_batch.drawCmdOffset + (i * consts::maxDrawsPerView) * 
                                               sizeof(shader::DrawCmd),
                                           view_batch.drawBuffer.buffer,
                                           count_offset, 
                                           consts::maxDrawsPerView,
                                           sizeof(shader::DrawCmd));
    }

    dev.dt.cmdEndRenderingKHR(draw_cmd);
}

static void issueDeferred(vk::Device &dev,
                          PipelineMP<1> &pipeline,
                          VkCommandBuffer draw_cmd,
                          BatchFrame &batch_frame,
                          VkExtent2D render_dims,
                          uint32_t total_num_views,
                          VkDescriptorSet asset_set,
                          VkDescriptorSet asset_mat_tex_set,
                          VkDescriptorSet index_buffer_set,
                          VkDescriptorSet pbr_set,
                          uint32_t view_width,
                          uint32_t view_height) 
{
    (void)asset_set;
    (void)asset_mat_tex_set;

    uint32_t max_image_dim_x = std::min(consts::maxTextureDim, consts::maxNumImagesX * view_width);
    uint32_t max_image_dim_y = std::min(consts::maxTextureDim, consts::maxNumImagesY * view_height);

    uint32_t max_images_x = max_image_dim_x / view_width;
    uint32_t max_images_y = max_image_dim_y / view_height;

    // The output buffer has been transitioned to general at the start of the frame.
    // The viz buffers have been transitioned to general before this happens.
    dev.dt.cmdBindPipeline(draw_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.hdls[0]);

    shader::DeferredLightingPushConstBR push_const = {
        .maxImagesXPerTarget = max_images_x,
        .maxImagesYPerTarget = max_images_y,
        .viewWidth = view_width,
        .viewHeight = view_height
    };

    dev.dt.cmdPushConstants(draw_cmd, pipeline.layout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0,
            sizeof(shader::DeferredLightingPushConstBR),
            &push_const);

    std::array draw_descriptors = {
        batch_frame.targetsSetLighting,
        index_buffer_set,
        batch_frame.viewInstanceSetLighting,
        pbr_set
    };

    dev.dt.cmdBindDescriptorSets(draw_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 pipeline.layout, 0,
                                 draw_descriptors.size(),
                                 draw_descriptors.data(),
                                 0, nullptr);

    uint32_t num_workgroups_x = utils::divideRoundUp(
        render_dims.width, 32_u32);
    uint32_t num_workgroups_y = utils::divideRoundUp(
        render_dims.height, 32_u32);
    uint32_t num_workgroups_z = total_num_views;

    dev.dt.cmdDispatch(
        draw_cmd, num_workgroups_x, num_workgroups_y, num_workgroups_z);
}

////////////////////////////////////////////////////////////////////////////////
// BATCH RENDERER PROTOTYPE IMPLEMENTATION                                    //
////////////////////////////////////////////////////////////////////////////////

struct BatchRenderer::Impl {
    vk::Device &dev;
    bool depthOnly;
    vk::MemoryAllocator &mem;

    uint32_t maxNumViews;
    uint32_t numWorlds;

    // Resources used in/for rendering the batch output
    // We use anything from double, triple, or whatever we can buffering to save
    // on memory usage

    // Required whether we do batch rendering or not
    PipelineMP<1> prepareViews;

    Optional<PipelineMP<1>> batchDraw;
    Optional<PipelineMP<1>> createVisualization;
    Optional<PipelineMP<1>> lighting;

    //One frame is on simulation frame
    HeapArray<BatchFrame> batchFrames;

    VkDescriptorSet assetSetPrepare;
    VkDescriptorSet assetSetDraw;
    VkDescriptorSet assetSetTextureMat;
    VkDescriptorSet assetSetLighting;

    VkExtent2D renderExtent;

    uint32_t selectedView;

    uint32_t currentFrame;

    VkQueue renderQueue;

    VkQueryPool timeQueryPool;
    uint64_t timestamps[2];

    std::vector<float> recordedTimings;

    HeapArray<VkRect2D> rects;
    HeapArray<VkViewport> viewports;


    // This pipeline prepares the draw commands in the buffered draw cmds buffer
    // Pipeline<1> prepareViews;

    Impl(const Config &cfg, RenderContext &rctx);
};

static const char *getDrawDeferredPath(bool render_rgb)
{
    if (render_rgb) {
        return "draw_deferred_rgb.hlsl";
    } else {
        return "draw_deferred_depth.hlsl";
    }
}

BatchRenderer::Impl::Impl(const Config &cfg,
                          RenderContext &rctx)
    : dev(rctx.dev),
      depthOnly(cfg.renderMode == RenderManager::Config::RenderMode::Depth),
      mem(rctx.alloc),
      maxNumViews(cfg.numWorlds * cfg.maxViewsPerWorld),
      numWorlds(cfg.numWorlds),
      // This is required whether we want the batch renderer or not
      prepareViews(makeComputePipeline(dev, rctx.pipelineCache, 4,
          sizeof(shader::PrepareViewPushConstant),
          4 + consts::numDrawCmdBuffers, rctx.repeatSampler,
          "prepare_views.hlsl", false, "main", makeShaders)),
      batchDraw(cfg.enableBatchRenderer ? 
          makeDrawPipeline(dev, rctx.pipelineCache, VK_NULL_HANDLE, 
                           consts::numDrawCmdBuffers * cfg.numFrames, 2, depthOnly, rctx.repeatSampler) :
          Optional<PipelineMP<1>>::none()),
      createVisualization(cfg.enableBatchRenderer ?
          makeComputePipeline(
              dev, rctx.pipelineCache, 1,
              sizeof(uint32_t) * 2,
              cfg.numFrames * consts::numDrawCmdBuffers, rctx.repeatSampler,
              "visualize_tris.hlsl", false, "visualize", makeShaders) :
          Optional<PipelineMP<1>>::none()),
      lighting(cfg.enableBatchRenderer ?
          makeComputePipeline(dev, rctx.pipelineCache, 4, 
              sizeof(shader::DeferredLightingPushConstBR),
              consts::numDrawCmdBuffers * cfg.numFrames, rctx.repeatSampler, 
              getDrawDeferredPath(!depthOnly), depthOnly, "lighting", makeShadersLighting) :
          Optional<PipelineMP<1>>::none()),
      batchFrames(cfg.numFrames),
      assetSetPrepare(rctx.asset_set_cull_),
      assetSetDraw(rctx.asset_set_draw_),
      assetSetTextureMat(rctx.asset_set_tex_compute_),
      assetSetLighting(rctx.asset_batch_lighting_set_),
      renderExtent { cfg.renderWidth, cfg.renderHeight },
      selectedView(0),
      currentFrame(0),
      renderQueue(rctx.renderQueue),
      rects(dev.maxViewports),
      viewports(dev.maxViewports)
{
    for (uint32_t i = 0; i < cfg.numFrames; i++) {
        makeBatchFrame(dev, &batchFrames[i], mem, cfg,
                       prepareViews,
                       batchDraw,
                       cfg.enableBatchRenderer ? lighting->descPools[0].makeSet() : VK_NULL_HANDLE,
                       cfg.enableBatchRenderer ? lighting->descPools[3].makeSet() : VK_NULL_HANDLE,
                       cfg.enableBatchRenderer,
                       rctx,
                       cfg.renderWidth, cfg.renderHeight,
                       depthOnly);
    }

    VkQueryPoolCreateInfo pool_create_info = {};
    pool_create_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    pool_create_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    pool_create_info.queryCount = 2;

    REQ_VK(dev.dt.createQueryPool(dev.hdl, &pool_create_info, 
                                  nullptr, &timeQueryPool));

    for (uint32_t i = 0; i < dev.maxViewports; ++i) {
        uint32_t x_start = i * cfg.renderWidth;

        rects[i] = VkRect2D {
            .offset = { (int32_t)x_start, 0 },
            .extent = { cfg.renderWidth, cfg.renderHeight }
        };

        viewports[i] = VkViewport {
            .x = (float)x_start,
            .y = 0.f,
            .width = (float)cfg.renderWidth,
            .height = (float)cfg.renderHeight,
            .minDepth = 0.f,
            .maxDepth = 1.f
        };
    }
}

BatchRenderer::BatchRenderer(const Config &cfg,
                             RenderContext &rctx)
    : impl(std::make_unique<Impl>(cfg, rctx)),
      didRender(false)
{}

BatchRenderer::~BatchRenderer()
{
    impl->dev.dt.destroyPipeline(impl->dev.hdl, impl->prepareViews.hdls[0], nullptr);
    impl->dev.dt.destroyPipelineLayout(impl->dev.hdl, impl->prepareViews.layout, nullptr);

    // If the batch renderer was enabled
    if (impl->batchDraw.has_value()) {
        impl->dev.dt.destroyPipeline(impl->dev.hdl, impl->batchDraw->hdls[0], nullptr);
        impl->dev.dt.destroyPipelineLayout(impl->dev.hdl, impl->batchDraw->layout, nullptr);

        impl->dev.dt.destroyPipeline(impl->dev.hdl, impl->createVisualization->hdls[0], nullptr);
        impl->dev.dt.destroyPipelineLayout(impl->dev.hdl, impl->createVisualization->layout, nullptr);

        impl->dev.dt.destroyPipeline(impl->dev.hdl, impl->lighting->hdls[0], nullptr);
        impl->dev.dt.destroyPipelineLayout(impl->dev.hdl, impl->lighting->layout, nullptr);

        for(int i=0;i<impl->batchFrames.size();i++){
            impl->dev.dt.destroyCommandPool(impl->dev.hdl, impl->batchFrames[i].prepareCmdPool, nullptr);
            impl->dev.dt.destroyCommandPool(impl->dev.hdl, impl->batchFrames[i].renderCmdPool, nullptr);
            impl->dev.dt.destroySemaphore(impl->dev.hdl, impl->batchFrames[i].prepareFinished, nullptr);
            impl->dev.dt.destroySemaphore(impl->dev.hdl, impl->batchFrames[i].renderFinished, nullptr);
            impl->dev.dt.destroySemaphore(impl->dev.hdl, impl->batchFrames[i].layoutTransitionFinished, nullptr);
            impl->dev.dt.destroyFence(impl->dev.hdl, impl->batchFrames[i].prepareFence, nullptr);
            impl->dev.dt.destroyFence(impl->dev.hdl, impl->batchFrames[i].renderFence, nullptr);

            for(int i2=0;i2<impl->batchFrames[i].targets.size();i2++){
                impl->dev.dt.destroyImageView(impl->dev.hdl, impl->batchFrames[i].targets[i2].vizBufferView, nullptr);
                impl->dev.dt.destroyImageView(impl->dev.hdl, impl->batchFrames[i].targets[i2].depthView, nullptr);
            }
        }
    }
    else {
        for(int i=0;i<impl->batchFrames.size();i++){
            impl->dev.dt.destroyCommandPool(impl->dev.hdl, impl->batchFrames[i].prepareCmdPool, nullptr);
            impl->dev.dt.destroyCommandPool(impl->dev.hdl, impl->batchFrames[i].renderCmdPool, nullptr);
            impl->dev.dt.destroySemaphore(impl->dev.hdl, impl->batchFrames[i].prepareFinished, nullptr);
            impl->dev.dt.destroySemaphore(impl->dev.hdl, impl->batchFrames[i].renderFinished, nullptr);
            impl->dev.dt.destroySemaphore(impl->dev.hdl, impl->batchFrames[i].layoutTransitionFinished, nullptr);
            impl->dev.dt.destroyFence(impl->dev.hdl, impl->batchFrames[i].prepareFence, nullptr);
            impl->dev.dt.destroyFence(impl->dev.hdl, impl->batchFrames[i].renderFence, nullptr);
        }
    }

    impl->dev.dt.destroyQueryPool(impl->dev.hdl, impl->timeQueryPool, nullptr);
}


static void issuePrepareViewsPipeline(vk::Device& dev,
                                      VkCommandBuffer& draw_cmd,
                                      PipelineMP<1>& prepare_views,
                                      BatchFrame& frame,
                                      DrawCommandPackage& batch,
                                      VkDescriptorSet& assetSetPrepareView,
                                      uint32_t num_worlds,
                                      uint32_t num_instances,
                                      uint32_t num_views,
                                      uint32_t view_start,
                                      uint32_t num_processed_batches,
                                      RenderContext &rctx)
{
    (void)num_views;
    (void)num_processed_batches;
    dev.dt.cmdBindPipeline(draw_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           prepare_views.hdls[0]);

    { // Dispatch the compute shader
        std::array view_gen_descriptors = {
            frame.viewInstanceSetPrepare,
            batch.drawBufferSetPrepare,
            assetSetPrepareView,
            // frame.viewAABBSetPrepare,
            rctx.loaded_assets_[0].aabbSet
        };

        dev.dt.cmdBindDescriptorSets(draw_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                     prepare_views.layout, 0,
                                     view_gen_descriptors.size(),
                                     view_gen_descriptors.data(),
                                     0, nullptr);

        shader::PrepareViewPushConstant view_push_const = {
            num_views, view_start, num_worlds, num_instances,
            consts::maxDrawsPerView
        };

        dev.dt.cmdPushConstants(draw_cmd, prepare_views.layout,
                                VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                sizeof(shader::PrepareViewPushConstant),
                                &view_push_const);

        uint32_t num_workgroups = num_views;
        dev.dt.cmdDispatch(draw_cmd, num_workgroups, 1, 1);
    }
}

static void issueMemoryBarrier(vk::Device &dev,
                               VkCommandBuffer draw_cmd,
                               VkAccessFlags src_access,
                               VkAccessFlags dst_access,
                               VkPipelineStageFlags src_stage,
                               VkPipelineStageFlags dst_stage)
{
    VkMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = src_access,
        .dstAccessMask = dst_access
    };

    dev.dt.cmdPipelineBarrier(draw_cmd, src_stage, dst_stage, 0, 1, &barrier, 
                              0, nullptr, 0, nullptr);
}

static void sortInstancesAndViewsCPU(EngineInterop *interop)
{
    for (uint32_t i = 0; i < *interop->bridge.totalNumInstances; ++i) {
        interop->iotaArrayInstancesCPU[i] = i;
    }

    for (uint32_t i = 0; i < *interop->bridge.totalNumViews; ++i) {
        interop->iotaArrayViewsCPU[i] = i;
    }

    { // Sort the indices based on worldID/entityID number
        std::sort(interop->iotaArrayInstancesCPU, 
                  interop->iotaArrayInstancesCPU + *interop->bridge.totalNumInstances,
                  [&interop] (uint32_t a, uint32_t b) {
                      return interop->bridge.instancesWorldIDs[a] < interop->bridge.instancesWorldIDs[b];
                  });

        std::sort(interop->iotaArrayViewsCPU,
                  interop->iotaArrayViewsCPU + *interop->bridge.totalNumViews,
                  [&interop] (uint32_t a, uint32_t b) {
                      return interop->bridge.viewsWorldIDs[a] < interop->bridge.viewsWorldIDs[b];
                  });
    }

    InstanceData *instances = (InstanceData *)interop->instancesCPU->ptr;
    PerspectiveCameraData *views = (PerspectiveCameraData *)interop->viewsCPU->ptr;

    { // Write the sorted array of views and instances
        for (uint32_t i = 0; i < *interop->bridge.totalNumInstances; ++i) {
            instances[i] = interop->bridge.instances[interop->iotaArrayInstancesCPU[i]];

            // We also need to have the sorted instance IDs in order to extract the offsets
            interop->sortedInstanceWorldIDs[i] = 
                interop->bridge.instancesWorldIDs[interop->iotaArrayInstancesCPU[i]];
        }

        for (uint32_t i = 0; i < *interop->bridge.totalNumViews; ++i) {
            views[i] = interop->bridge.views[interop->iotaArrayViewsCPU[i]];

            interop->sortedViewWorldIDs[i] = 
                interop->bridge.viewsWorldIDs[interop->iotaArrayViewsCPU[i]];
        }
    }
}

static void sortLightsCPU(EngineInterop *interop)
{
    for (uint32_t i = 0; i < *interop->bridge.totalNumLights; ++i) {
        interop->iotaArrayLightOffsetsCPU[i] = i;
    }

    std::sort(interop->iotaArrayLightOffsetsCPU,
                interop->iotaArrayLightOffsetsCPU + *interop->bridge.totalNumLights,
                [&interop] (uint32_t a, uint32_t b) {
                    return interop->bridge.lightOffsets[a] < interop->bridge.lightOffsets[b];
                });

    LightDesc *lights = (LightDesc *)interop->lightsCPU->ptr;

    for (uint32_t i = 0; i < *interop->bridge.totalNumLights; ++i) {
        lights[i] = interop->bridge.lights[interop->iotaArrayLightOffsetsCPU[i]];

        interop->sortedLightOffsets[i] = 
            interop->bridge.lightOffsets[interop->iotaArrayLightOffsetsCPU[i]];
    }
}

static void computeInstanceOffsets(EngineInterop *interop, uint32_t num_worlds)
{
    uint32_t *instanceOffsets = (uint32_t *)interop->instanceOffsetsCPU->ptr;

    for (int i = 0; i < (int)num_worlds; ++i) {
        instanceOffsets[i] = 0;
    }

    for (uint32_t i = 1; i < *interop->bridge.totalNumInstances; ++i) {
        uint32_t current_world_id = (uint32_t)(interop->sortedInstanceWorldIDs[i] >> 32);
        uint32_t prev_world_id = (uint32_t)(interop->sortedInstanceWorldIDs[i-1] >> 32);

        if (current_world_id != prev_world_id) {
            instanceOffsets[current_world_id] = i;
        }
    }
}

static void computeViewOffsets(EngineInterop *interop, uint32_t num_worlds)
{
    uint32_t *viewOffsets = (uint32_t *)interop->viewOffsetsCPU->ptr;

    for (int i = 0; i < (int)num_worlds; ++i) {
        viewOffsets[i] = 0;
    }

    for (uint32_t i = 1; i < *interop->bridge.totalNumViews; ++i) {
        uint32_t current_world_id = (uint32_t)(interop->sortedViewWorldIDs[i] >> 32);
        uint32_t prev_world_id = (uint32_t)(interop->sortedViewWorldIDs[i-1] >> 32);

        if (current_world_id != prev_world_id) {
            viewOffsets[current_world_id] = i;
        }
    }
}

static void computeLightOffsets(EngineInterop *interop, uint32_t num_worlds)
{
    uint32_t *lightOffsets = (uint32_t *)interop->lightOffsetsCPU->ptr;

    for (int i = 0; i < (int)num_worlds; ++i) {
        lightOffsets[i] = 0;
    }

    for (uint32_t i = 1; i < *interop->bridge.totalNumLights; ++i) {
        uint32_t current_world_id = (uint32_t)(interop->sortedLightOffsets[i] >> 32);
        uint32_t prev_world_id = (uint32_t)(interop->sortedLightOffsets[i-1] >> 32);

        if (current_world_id != prev_world_id) {
            lightOffsets[current_world_id] = i;
        }
    }
}

void BatchRenderer::prepareForRendering(BatchRenderInfo info,
                                        EngineInterop *interop)
{
    // Circles between 0 to number of frames (not anymore, there is only one frame now)
    uint32_t frame_index = impl->currentFrame;

    { // Flush CPU buffers if we used CPU buffers
        if (interop->viewsCPU.has_value()) {
            *interop->bridge.totalNumViews = interop->bridge.totalNumViewsCPUInc->load_acquire();
            *interop->bridge.totalNumInstances = interop->bridge.totalNumInstancesCPUInc->load_acquire();

            info.numInstances = *interop->bridge.totalNumInstances;
            info.numViews = *interop->bridge.totalNumViews;

            interop->bridge.totalNumViewsCPUInc->store_release(0);
            interop->bridge.totalNumInstancesCPUInc->store_release(0);

            // First, need to perform the sorts
            sortInstancesAndViewsCPU(interop);
            computeInstanceOffsets(interop, info.numWorlds);
            computeViewOffsets(interop, info.numWorlds);

            // Need to flush engine input state before copy
            interop->viewsCPU->flush(impl->dev);
            interop->viewOffsetsCPU->flush(impl->dev);
            interop->instancesCPU->flush(impl->dev);
            interop->instanceOffsetsCPU->flush(impl->dev);
            interop->aabbCPU->flush(impl->dev);
        }

        if (interop->voxelInputCPU.has_value()) {
            // Need to flush engine input state before copy
            interop->voxelInputCPU->flush(impl->dev);
        }

        if (interop->lightsCPU.has_value()) {
            *interop->bridge.totalNumLights = interop->bridge.totalNumLightsCPUInc->load_acquire();

            info.numLights = *interop->bridge.totalNumLights;

            interop->bridge.totalNumLightsCPUInc->store_release(0);

            // First, need to perform the sorts
            sortLightsCPU(interop);
            computeLightOffsets(interop, info.numWorlds);

            interop->lightsCPU->flush(impl->dev);
            interop->lightOffsetsCPU->flush(impl->dev);
        }
    }

    BatchFrame &frame_data = impl->batchFrames[frame_index];

    { // Wait for the frame to be ready
        if (frame_data.latestOp != LatestOperation::None) {
            impl->dev.dt.waitForFences(impl->dev.hdl, 1, 
                                       &frame_data.getLatestFence(),
                                       VK_TRUE, UINT64_MAX);
        }
    }

    BatchImportedBuffers &batch_buffers = getImportedBuffers(frame_index);

    // Start the command buffer and stuff
    VkCommandBuffer draw_cmd = frame_data.prepareCmdbuf;
    {
        REQ_VK(impl->dev.dt.resetCommandPool(impl->dev.hdl, frame_data.prepareCmdPool, 0));
        VkCommandBufferBeginInfo begin_info {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        REQ_VK(impl->dev.dt.beginCommandBuffer(draw_cmd, &begin_info));
    }

    { // Import the views
        VkDeviceSize num_views_bytes = info.numViews *
            sizeof(shader::PackedViewData);

        VkBufferCopy view_data_copy = {
            .srcOffset = 0, .dstOffset = 0,
            .size = num_views_bytes
        };

       impl->dev.dt.cmdCopyBuffer(draw_cmd, interop->viewsHdl,
                             batch_buffers.views.buffer,
                             1, &view_data_copy);
    }

    { // Import the instances
        VkDeviceSize num_instances_bytes = info.numInstances *
            sizeof(shader::PackedInstanceData);

        VkBufferCopy instance_data_copy = {
            .srcOffset = 0, .dstOffset = 0,
            .size = num_instances_bytes
        };

        impl->dev.dt.cmdCopyBuffer(draw_cmd, interop->instancesHdl,
                             batch_buffers.instances.buffer,
                             1, &instance_data_copy);
    }

    { // Import the offsets for instances
        VkDeviceSize num_offsets_bytes = info.numWorlds *
            sizeof(int32_t);

        VkBufferCopy offsets_data_copy = {
            .srcOffset = 0, .dstOffset = 0,
            .size = num_offsets_bytes
        };

        impl->dev.dt.cmdCopyBuffer(draw_cmd, interop->instanceOffsetsHdl,
                             batch_buffers.instanceOffsets.buffer,
                             1, &offsets_data_copy);
    }

#if 0
    { // Import the aabbs for instances
        VkDeviceSize num_aabbs_bytes = info.numInstances *
            sizeof(shader::AABB);

        VkBufferCopy aabb_data_copy = {
            .srcOffset = 0, .dstOffset = 0,
            .size = num_aabbs_bytes
        };

        impl->dev.dt.cmdCopyBuffer(draw_cmd, interop->aabbHdl,
                             batch_buffers.aabbs.buffer,
                             1, &aabb_data_copy);
    }
#endif

    { // Import the offsets for views
        VkDeviceSize num_offsets_bytes = info.numWorlds *
            sizeof(int32_t);

        VkBufferCopy offsets_data_copy = {
            .srcOffset = 0, .dstOffset = 0,
            .size = num_offsets_bytes
        };

        impl->dev.dt.cmdCopyBuffer(draw_cmd, interop->viewOffsetsHdl,
                             batch_buffers.viewOffsets.buffer,
                             1, &offsets_data_copy);
    }

    { // Import the lights
        VkDeviceSize num_lights_bytes = info.numLights *
            sizeof(shader::LightDesc);

        VkBufferCopy lights_data_copy = {
            .srcOffset = 0, .dstOffset = 0,
            .size = num_lights_bytes
        };

        impl->dev.dt.cmdCopyBuffer(draw_cmd, interop->lightsHdl,
                             batch_buffers.lights.buffer,
                             1, &lights_data_copy);
    }

    { // Import the light offsets
        VkDeviceSize num_offsets_bytes = info.numWorlds *
            sizeof(int32_t);

        VkBufferCopy offsets_data_copy = {
            .srcOffset = 0, .dstOffset = 0,
            .size = num_offsets_bytes
        };

        impl->dev.dt.cmdCopyBuffer(draw_cmd, interop->lightOffsetsHdl,
                             batch_buffers.lightOffsets.buffer,
                             1, &offsets_data_copy);
    }

    REQ_VK(impl->dev.dt.endCommandBuffer(draw_cmd));

    VkSemaphore sema_wait = VK_NULL_HANDLE;
    VkPipelineStageFlags wait_flags = VK_PIPELINE_STAGE_TRANSFER_BIT;

    if (didRender) {
        sema_wait = frame_data.renderFinished;
    }

    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreCount = (sema_wait != VK_NULL_HANDLE) ? 1u : 0u,
        .pWaitSemaphores = &sema_wait,
        .pWaitDstStageMask = &wait_flags,
        .commandBufferCount = 1,
        .pCommandBuffers = &draw_cmd,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &frame_data.prepareFinished
    };

    REQ_VK(impl->dev.dt.resetFences(impl->dev.hdl, 1, &frame_data.prepareFence));
    REQ_VK(impl->dev.dt.queueSubmit(impl->renderQueue, 1, &submit_info, frame_data.prepareFence));

    frame_data.latestOp = LatestOperation::RenderPrepare;

    didRender = true;
}

static void packSky( const vk::Device &dev,
                     vk::HostBuffer &staging)
{
    render::shader::SkyData *data = (render::shader::SkyData *)staging.ptr;

    /* 
       Based on the values of Eric Bruneton's atmosphere model.
       We aren't going to calculate these manually come on (at least not yet)
    */

    /* 
       These are the irradiance values at the top of the Earth's atmosphere
       for the wavelengths 680nm (red), 550nm (green) and 440nm (blue) 
       respectively. These are in W/m2
       */
    data->solarIrradiance = math::Vector4{1.474f, 1.8504f, 1.91198f, 0.0f};
    // Angular radius of the Sun (radians)
    data->solarAngularRadius = 0.004675f;
    data->bottomRadius = 6360.0f / 2.0f;
    data->topRadius = 6420.0f / 2.0f;

    data->rayleighDensity.layers[0] =
        render::shader::DensityLayer { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, {} };
    data->rayleighDensity.layers[1] =
        render::shader::DensityLayer { 0.0f, 1.0f, -0.125f, 0.0f, 0.0f, {} };
    data->rayleighScatteringCoef =
        math::Vector4{0.005802f, 0.013558f, 0.033100f, 0.0f};

    data->mieDensity.layers[0] =
        render::shader::DensityLayer { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, {} };
    data->mieDensity.layers[1] =
        render::shader::DensityLayer { 0.0f, 1.0f, -0.833333f, 0.0f, 0.0f, {} };
    data->mieScatteringCoef = math::Vector4{0.003996f, 0.003996f, 0.003996f, 0.0f};
    data->mieExtinctionCoef = math::Vector4{0.004440f, 0.004440f, 0.004440f, 0.0f};

    data->miePhaseFunctionG = 0.8f;

    data->absorptionDensity.layers[0] =
        render::shader::DensityLayer { 25.000000f, 0.000000f, 0.000000f, 0.066667f, -0.666667f, {} };
    data->absorptionDensity.layers[1] =
        render::shader::DensityLayer { 0.000000f, 0.000000f, 0.000000f, -0.066667f, 2.666667f, {} };
    data->absorptionExtinctionCoef =
        math::Vector4{0.000650f, 0.001881f, 0.000085f, 0.0f};
    data->groundAlbedo = math::Vector4{0.050000f, 0.050000f, 0.050000f, 0.0f};
    data->muSunMin = -0.207912f;
    data->wPlanetCenter =
      math::Vector4{0.0f, 0.0f, -data->bottomRadius, 0.0f};
    data->sunSize = math::Vector4{
            0.0046750340586467079f, 0.99998907220740285f, 0.0f, 0.0f};

    staging.flush(dev);
}


void BatchRenderer::renderViews(BatchRenderInfo info,
                                const DynArray<AssetData> &loaded_assets,
                                EngineInterop *interop,
                                RenderContext &rctx)
{
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    (void)interop;

    // prepareForRendering(info, interop);

    // Circles between 0 to number of frames (not anymore, there is only one frame now)
    uint32_t frame_index = impl->currentFrame;

    BatchFrame &frame_data = impl->batchFrames[frame_index];

    // Start the command buffer and stuff
    VkCommandBuffer draw_cmd = frame_data.renderCmdbuf;
    {
        REQ_VK(impl->dev.dt.resetCommandPool(impl->dev.hdl, frame_data.renderCmdPool, 0));
        VkCommandBufferBeginInfo begin_info {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        REQ_VK(impl->dev.dt.beginCommandBuffer(draw_cmd, &begin_info));
    }

    impl->dev.dt.cmdResetQueryPool(draw_cmd, impl->timeQueryPool, 0, 2);

    impl->dev.dt.cmdWriteTimestamp(draw_cmd, 
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, impl->timeQueryPool, 0);

    ////////////////////////////////////////////////////////////////

    { // Import sky and lighting information first
    /*
        packLighting(impl->dev, frame_data.lightingStaging, rctx.lights_);
        VkBufferCopy light_copy {
            .srcOffset = 0,
            .dstOffset = 0,
            .size = sizeof(render::shader::LightDesc) * InternalConfig::maxLights
        };
        impl->dev.dt.cmdCopyBuffer(draw_cmd, frame_data.lightingStaging.buffer,
                             frame_data.lighting.buffer,
                             1, &light_copy);
    */

        packSky(impl->dev, frame_data.skyInputStaging);
        VkBufferCopy sky_copy {
            .srcOffset = 0,
            .dstOffset = 0,
            .size = sizeof(render::shader::SkyData)
        };
        impl->dev.dt.cmdCopyBuffer(draw_cmd, frame_data.skyInputStaging.buffer,
                             frame_data.skyInput.buffer,
                             1, &sky_copy);
    }

    { // Prepare memory written to by ECS with barrier
        std::array barriers = {
            VkBufferMemoryBarrier{
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                frame_data.buffers.views.buffer,
                0, VK_WHOLE_SIZE
            },
            VkBufferMemoryBarrier{
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                frame_data.buffers.instances.buffer,
                0, VK_WHOLE_SIZE
            },
            VkBufferMemoryBarrier{
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                frame_data.buffers.instanceOffsets.buffer,
                0, VK_WHOLE_SIZE
            },
            VkBufferMemoryBarrier{
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                frame_data.lighting.buffer,
                0, VK_WHOLE_SIZE
            },
            VkBufferMemoryBarrier{
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                frame_data.skyInput.buffer,
                0, VK_WHOLE_SIZE
            },
        };

        impl->dev.dt.cmdPipelineBarrier(
            draw_cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, barriers.size(), barriers.data(),
            0, nullptr);

        for (int i = 0; i < (int)consts::numDrawCmdBuffers; ++i) {
            auto &draw_pckg = frame_data.drawPackageSwapchain[i];

            impl->dev.dt.cmdFillBuffer(draw_cmd, draw_pckg.drawBuffer.buffer, 
                0, sizeof(uint32_t) * draw_pckg.numDrawCounts, 0);
        }
    }

    // Issue the memory barrier for the draw packages
    issueMemoryBarrier(impl->dev, draw_cmd,
        VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT |
            VK_ACCESS_SHADER_READ_BIT,
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT |
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    uint32_t num_processed_views = 0;

    // Loop over all the layered targets
    for (uint32_t target_idx = 0;
            target_idx < frame_data.targets.size();
            ++target_idx) {
        auto &target = frame_data.targets[target_idx];

        uint32_t draw_package_idx = target_idx % consts::numDrawCmdBuffers;

        auto &draw_package =
            frame_data.drawPackageSwapchain[draw_package_idx];

        { // Issue buffer barrier for this draw package buffer
            VkBufferMemoryBarrier draw_pckg_barrier = {
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                    VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT |
                    VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
                VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                draw_package.drawBuffer.buffer,
                0, VK_WHOLE_SIZE
            };

            impl->dev.dt.cmdPipelineBarrier(
                draw_cmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT | 
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                    VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 0, nullptr, 1, &draw_pckg_barrier,
                0, nullptr);
        }

        // Issue the prepare views pipeline with the current draw package
        issuePrepareViewsPipeline(impl->dev, draw_cmd,
                                  impl->prepareViews,
                                  frame_data,
                                  draw_package,
                                  impl->assetSetPrepare,
                                  info.numWorlds,
                                  info.numInstances,
                                  target.numViews,
                                  num_processed_views,
                                  draw_package_idx,
                                  rctx);

        { // Issue buffer barrier for this draw package buffer
            VkBufferMemoryBarrier draw_pckg_barrier = {
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
                VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                draw_package.drawBuffer.buffer,
                0, VK_WHOLE_SIZE
            };

            impl->dev.dt.cmdPipelineBarrier(
                draw_cmd,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
               VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                   VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT |
                   VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0, 0, nullptr, 1, &draw_pckg_barrier,
                0, nullptr);
        }

        
        // Now, start the rasterization
        issueRasterLayoutTransitions(impl->dev,
                target,
                draw_cmd);

        // Begin rendering
        issueRasterization(impl->dev,
                           *(impl->batchDraw),
                           target,
                           draw_cmd,
                           draw_package,
                           frame_data,
                           impl->assetSetLighting,
                           impl->assetSetTextureMat,
                           impl->renderExtent,
                           loaded_assets,
                           impl->depthOnly);

        issueComputeLayoutTransitions(impl->dev,
                               target,
                               draw_cmd);

        issueMemoryBarrier(impl->dev, draw_cmd,
                           VK_ACCESS_SHADER_READ_BIT,
                           VK_ACCESS_TRANSFER_WRITE_BIT,
                           VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                               VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT |
                               VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT);

        impl->dev.dt.cmdFillBuffer(draw_cmd, draw_package.drawBuffer.buffer,
                                   0, target.numViews * sizeof(uint32_t), 0);

        num_processed_views += target.numViews;
    }

    issueDeferred(
        impl->dev,
        *(impl->lighting), 
        draw_cmd,
        frame_data,
        impl->renderExtent,
        info.numViews,
        impl->assetSetLighting,
        impl->assetSetTextureMat,
        loaded_assets[0].indexBufferSet,
        frame_data.pbrSet,
        frame_data.targets[0].viewWidth,
        frame_data.targets[0].viewHeight);

    impl->dev.dt.cmdWriteTimestamp(draw_cmd, 
                VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, impl->timeQueryPool, 1);

    // End the command buffer and stuff
    REQ_VK(impl->dev.dt.endCommandBuffer(draw_cmd));

    VkPipelineStageFlags prepare_wait_flag =
        VK_PIPELINE_STAGE_TRANSFER_BIT;

    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &frame_data.prepareFinished,
        .pWaitDstStageMask = &prepare_wait_flag,
        .commandBufferCount = 1,
        .pCommandBuffers = &draw_cmd,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &frame_data.renderFinished
    };

    REQ_VK(impl->dev.dt.resetFences(impl->dev.hdl, 1, &frame_data.renderFence));
    REQ_VK(impl->dev.dt.queueSubmit(impl->renderQueue, 1, &submit_info, frame_data.renderFence));

    REQ_VK(impl->dev.dt.deviceWaitIdle(impl->dev.hdl));

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    double duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    float duration_ms = (float)(duration/1000000.0);

    impl->recordedTimings.push_back(duration_ms);

    impl->dev.dt.getQueryPoolResults(
                impl->dev.hdl, impl->timeQueryPool, 0, 2, sizeof(uint64_t) * 2, 
                impl->timestamps, sizeof(uint64_t),
                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

    //double delta_in_ms = double(impl->timestamps[1] - impl->timestamps[0]) *
    //    impl->dev.timestampPeriod / 1000000.0;

    // printf("rasterizer batch renderer took %f ms\n", (float)delta_in_ms);
    // impl->recordedTimings.push_back((float)delta_in_ms);

    frame_data.latestOp = LatestOperation::RenderViews;
}


#if 0
void BatchRenderer::renderViews(BatchRenderInfo info,
                                const DynArray<AssetData> &loaded_assets,
                                EngineInterop *interop,
                                RenderContext &rctx) 
{ 
    (void)interop;

    // prepareForRendering(info, interop);

    // Circles between 0 to number of frames (not anymore, there is only one frame now)
    uint32_t frame_index = impl->currentFrame;

    BatchFrame &frame_data = impl->batchFrames[frame_index];

    // Start the command buffer and stuff
    VkCommandBuffer draw_cmd = frame_data.renderCmdbuf;
    {
        REQ_VK(impl->dev.dt.resetCommandPool(impl->dev.hdl, frame_data.renderCmdPool, 0));
        VkCommandBufferBeginInfo begin_info {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        REQ_VK(impl->dev.dt.beginCommandBuffer(draw_cmd, &begin_info));
    }

    impl->dev.dt.cmdResetQueryPool(draw_cmd, impl->timeQueryPool, 0, 2);

    impl->dev.dt.cmdWriteTimestamp(draw_cmd, 
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, impl->timeQueryPool, 0);

    ////////////////////////////////////////////////////////////////

    { // Import sky and lighting information first
        packLighting(impl->dev, frame_data.lightingStaging, rctx.lights_);
        VkBufferCopy light_copy {
            .srcOffset = 0,
            .dstOffset = 0,
            .size = sizeof(render::shader::LightDesc) * InternalConfig::maxLights
        };
        impl->dev.dt.cmdCopyBuffer(draw_cmd, frame_data.lightingStaging.buffer,
                             frame_data.lighting.buffer,
                             1, &light_copy);

        packSky(impl->dev, frame_data.skyInputStaging);
        VkBufferCopy sky_copy {
            .srcOffset = 0,
            .dstOffset = 0,
            .size = sizeof(render::shader::SkyData)
        };
        impl->dev.dt.cmdCopyBuffer(draw_cmd, frame_data.skyInputStaging.buffer,
                             frame_data.skyInput.buffer,
                             1, &sky_copy);
    }

    { // Prepare memory written to by ECS with barrier
        std::array barriers = {
            VkBufferMemoryBarrier{
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                frame_data.buffers.views.buffer,
                0, VK_WHOLE_SIZE
            },
            VkBufferMemoryBarrier{
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                frame_data.buffers.instances.buffer,
                0, VK_WHOLE_SIZE
            },
            VkBufferMemoryBarrier{
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                frame_data.buffers.instanceOffsets.buffer,
                0, VK_WHOLE_SIZE
            },
            VkBufferMemoryBarrier{
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                frame_data.buffers.aabbs.buffer,
                0, VK_WHOLE_SIZE
            },
            VkBufferMemoryBarrier{
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                frame_data.lighting.buffer,
                0, VK_WHOLE_SIZE
            },
            VkBufferMemoryBarrier{
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                frame_data.skyInput.buffer,
                0, VK_WHOLE_SIZE
            },
        };

        impl->dev.dt.cmdPipelineBarrier(
            draw_cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, barriers.size(), barriers.data(),
            0, nullptr);

        for (int i = 0; i < (int)consts::numDrawCmdBuffers; ++i) {
            impl->dev.dt.cmdFillBuffer(draw_cmd, frame_data.drawPackageSwapchain[i].drawBuffer.buffer, 
                                       0, sizeof(uint32_t), 0);
        }
    }

    uint32_t num_views = info.numViews;

    struct BatchInfo {
        uint32_t numViews;
        uint32_t offset;
    };

    uint32_t views_per_batch = impl->dev.maxNumLayersPerImage *
                               impl->dev.maxViewports;

    uint32_t num_batches = utils::divideRoundUp(num_views, views_per_batch);
    HeapArray<BatchInfo> batch_infos(num_batches);

    { // Populate batch infos
        uint32_t views_left = num_views;
        for (int i = 0; i < (int)num_batches-1; ++i) {
            batch_infos[i].numViews = views_per_batch;
            views_left -= views_per_batch;
        }

        batch_infos[num_batches-1].numViews = views_left;
        batch_infos[0].offset = 0;

        for (int i = 1; i < (int)num_batches; ++i) {
            batch_infos[i].offset = batch_infos[i-1].numViews + batch_infos[i-1].offset;
        }
    }

    uint32_t num_iterations = utils::divideRoundUp(num_batches, consts::numDrawCmdBuffers);

    for (int iter = 0; iter < (int)num_iterations; ++iter) {
        int cur_num_batches = std::min(consts::numDrawCmdBuffers, 
                                       num_batches - iter * consts::numDrawCmdBuffers);

        issueMemoryBarrier(impl->dev, draw_cmd,
            VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                VK_ACCESS_SHADER_READ_BIT,
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT |
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        for (int batch = 0; batch < (int)cur_num_batches; ++batch) {
            int batch_no = batch + iter * consts::numDrawCmdBuffers;

            issuePrepareViewsPipeline(impl->dev, draw_cmd, 
                                      impl->prepareViews,
                                      frame_data, 
                                      frame_data.drawPackageSwapchain[batch],
                                      impl->assetSetPrepare,
                                      info.numWorlds, 
                                      info.numInstances,
                                      batch_infos[batch_no].numViews,
                                      batch_infos[batch_no].offset,
                                      batch_no);
        }

        issueMemoryBarrier(impl->dev, draw_cmd,
                           VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
                           VK_ACCESS_MEMORY_READ_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                               VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT |
                               VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

        for (int batch = 0; batch < (int)cur_num_batches; ++batch) {
            int batch_no = batch + iter * consts::numDrawCmdBuffers;
            
            issueRasterLayoutTransitions(impl->dev,
                                   frame_data.targets[batch_no],
                                   draw_cmd);
        }

        for (int batch = 0; batch < (int)cur_num_batches; ++batch) {
            int batch_no = batch + iter * consts::numDrawCmdBuffers;

            //Finish rest of draws for the frame
            issueRasterization(impl->dev,
                               *(impl->batchDraw),
                               frame_data.targets[batch_no],
                               draw_cmd,
                               frame_data.drawPackageSwapchain[batch],
                               frame_data,
                               impl->assetSetDraw,
                               impl->renderExtent,
                               loaded_assets,
                               impl->viewports.data(),
                               impl->rects.data());
        }

        for (int batch = 0; batch < (int)cur_num_batches; ++batch) {
            int batch_no = batch + iter * consts::numDrawCmdBuffers;
            
            issueComputeLayoutTransitions(impl->dev,
                                   frame_data.targets[batch_no],
                                   draw_cmd);
        }

        issueMemoryBarrier(impl->dev, draw_cmd,
                           VK_ACCESS_SHADER_READ_BIT,
                           VK_ACCESS_TRANSFER_WRITE_BIT,
                           VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                               VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT |
                               VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT);


        for (int batch = 0; batch < (int)cur_num_batches; ++batch) {
            impl->dev.dt.cmdFillBuffer(draw_cmd, frame_data.drawPackageSwapchain[batch].drawBuffer.buffer,
                                       0, sizeof(uint32_t), 0);
        }
    }

    issueDeferred(
        impl->dev,
        *(impl->lighting), 
        draw_cmd,
        frame_data,
        impl->renderExtent,
        num_views,
        impl->assetSetLighting,
        impl->assetSetTextureMat,
        loaded_assets[0].indexBufferSet,
        frame_data.pbrSet);

    impl->dev.dt.cmdWriteTimestamp(draw_cmd, 
                VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, impl->timeQueryPool, 1);

    // End the command buffer and stuff
    REQ_VK(impl->dev.dt.endCommandBuffer(draw_cmd));

    VkPipelineStageFlags prepare_wait_flag =
        VK_PIPELINE_STAGE_TRANSFER_BIT;

    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &frame_data.prepareFinished,
        .pWaitDstStageMask = &prepare_wait_flag,
        .commandBufferCount = 1,
        .pCommandBuffers = &draw_cmd,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &frame_data.renderFinished
    };

    REQ_VK(impl->dev.dt.resetFences(impl->dev.hdl, 1, &frame_data.renderFence));
    REQ_VK(impl->dev.dt.queueSubmit(impl->renderQueue, 1, &submit_info, frame_data.renderFence));

    impl->dev.dt.getQueryPoolResults(
                impl->dev.hdl, impl->timeQueryPool, 0, 2, sizeof(uint64_t) * 2, 
                impl->timestamps, sizeof(uint64_t),
                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

    float delta_in_ms = float(impl->timestamps[1] - impl->timestamps[0]) *
        impl->dev.timestampPeriod / 1000000.0f;

    // printf("rasterizer batch renderer took %f ms\n", delta_in_ms);
    impl->recordedTimings.push_back(delta_in_ms);

    frame_data.latestOp = LatestOperation::RenderViews;
}
#endif

BatchImportedBuffers &BatchRenderer::getImportedBuffers(uint32_t frame_id) 
{
    return impl->batchFrames[frame_id].buffers;
}

const vk::LocalBuffer & BatchRenderer::getRGBBuffer() const
{
    return impl->batchFrames[0].rgbOutput.buf;
}

const vk::LocalBuffer & BatchRenderer::getDepthBuffer() const
{
    return impl->batchFrames[0].depthOutput.buf;
}

// Get the semaphore that the viewer renderer has to wait on
VkSemaphore BatchRenderer::getLatestWaitSemaphore()
{
    uint32_t last_frame = (impl->currentFrame + impl->batchFrames.size() - 1) %
        impl->batchFrames.size();

    assert(impl->batchFrames[last_frame].latestOp != LatestOperation::None);

    if (impl->batchFrames[last_frame].latestOp == LatestOperation::RenderPrepare) {
        return impl->batchFrames[last_frame].prepareFinished;
    } else if (impl->batchFrames[last_frame].latestOp == LatestOperation::RenderViews) {
        return impl->batchFrames[last_frame].renderFinished;
    } else if (impl->batchFrames[last_frame].latestOp == LatestOperation::Transition) {
        return impl->batchFrames[last_frame].layoutTransitionFinished;
    }

    return VK_NULL_HANDLE;
}

const uint8_t * BatchRenderer::getRGBCUDAPtr() const
{
#ifndef MADRONA_VK_CUDA_SUPPORT
    return nullptr;
#else
    return (uint8_t *)impl->batchFrames[0].rgbOutputCUDA.getDevicePointer();
#endif
}

const float * BatchRenderer::getDepthCUDAPtr() const
{
#ifndef MADRONA_VK_CUDA_SUPPORT
    return nullptr;
#else
    return (float *)impl->batchFrames[0].depthOutputCUDA.getDevicePointer();
#endif
}

}
