#include "batch_proto.hpp"
#include "madrona/utils.hpp"
#include "madrona/viz/interop.hpp"
#include "viewer_renderer.hpp"
#include "shader.hpp"

#include "madrona/heap_array.hpp"

#include <array>
#include <filesystem>

#include "vk/memory.hpp"
#include "vulkan/vulkan_core.h"


using namespace madrona::render;
using madrona::render::vk::checkVk;

namespace madrona::viz {

namespace consts {
inline constexpr uint32_t maxDrawsPerLayeredImage = 65536;
inline constexpr VkFormat colorFormat = VK_FORMAT_R32G32_UINT;
inline constexpr VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;
inline constexpr VkFormat outputColorFormat = VK_FORMAT_R8G8B8A8_UNORM;
inline constexpr uint32_t numDrawCmdBuffers = 3; // Triple buffering
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
                                                   vk::MemoryAllocator &alloc
                                                   /*vk::FixedDescriptorPool &pool*/)
{
    uint32_t num_images = utils::divideRoundUp(max_num_views,
                                               dev.maxNumLayersPerImage);

    HeapArray<LayeredTarget> local_images (num_images);

    uint32_t layer_count = max_num_views;

    for (int i = 0; i < (int)num_images; ++i) {
        uint32_t current_layer_count = std::min(layer_count, dev.maxNumLayersPerImage);

        LayeredTarget target = {
            .vizBuffer = alloc.makeColorAttachment(width, height,
                                                   current_layer_count,
                                                   consts::colorFormat),
            .depth = alloc.makeDepthAttachment(width, height,
                                               current_layer_count,
                                               consts::depthFormat),
            .output = alloc.makeColorAttachment(width,height,
                                                current_layer_count,
                                                consts::outputColorFormat)
            // .lightingSet=pool.makeSet()
        };

        VkImageViewCreateInfo view_info = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = current_layer_count
            }
        };

        view_info.image = target.vizBuffer.image;
        view_info.format = consts::colorFormat;
        REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &target.vizBufferView));

        view_info.image = target.output.image;
        view_info.format = consts::outputColorFormat;
        REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &target.outputView));

        view_info.image = target.depth.image;
        view_info.format = consts::depthFormat;
        view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &target.depthView));

        target.layerCount = current_layer_count;

#if 0
        std::array<VkWriteDescriptorSet, 2> desc_updates;
        VkDescriptorImageInfo vis_buffer_info;
        vis_buffer_info.imageView = target.vizBufferView;
        vis_buffer_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        vis_buffer_info.sampler = VK_NULL_HANDLE;

        vk::DescHelper::storageImage(desc_updates[0], target.lightingSet, &vis_buffer_info, 0);

        VkDescriptorImageInfo output_info;
        output_info.imageView = target.outputView;
        output_info.imageLayout =  VK_IMAGE_LAYOUT_GENERAL;
        output_info.sampler = VK_NULL_HANDLE;

        vk::DescHelper::storageImage(desc_updates[1], target.lightingSet, &output_info, 1);
        vk::DescHelper::update(dev, desc_updates.data(), desc_updates.size());
#endif

        local_images.emplace(i, std::move(target));

        layer_count -= dev.maxNumLayersPerImage;
    }

    return local_images;
}



////////////////////////////////////////////////////////////////////////////////
// DRAW COMMAND BUFFER CREATION                                               //
////////////////////////////////////////////////////////////////////////////////
struct DrawCommandPackage {
    // Contains parameters to an actual vulkan draw command
    vk::LocalBuffer drawCmds;
    // Contains information about the object being drawn (instance and material)
    vk::LocalBuffer drawInfos;
};

static HeapArray<vk::LocalBuffer> makeDrawCmdBuffers(uint32_t num_draw_cmd_buffers,
                                                    const vk::Device &dev,
                                                    vk::MemoryAllocator &alloc)
{
    (void)dev;
    HeapArray<vk::LocalBuffer> buffers (num_draw_cmd_buffers);

    for (uint32_t i = 0; i < num_draw_cmd_buffers; ++i) {
        VkDeviceSize buffer_size = sizeof(shader::DrawCmd) * 
                                   consts::maxDrawsPerLayeredImage;

        buffers.emplace(i, alloc.makeLocalBuffer(buffer_size).value());
    }

    return buffers;
}



////////////////////////////////////////////////////////////////////////////////
// RENDER PIPELINE CREATION                                                   //
////////////////////////////////////////////////////////////////////////////////
static vk::PipelineShaders makeDrawShaders(const vk::Device &dev, 
                                           VkSampler repeat_sampler,
                                           VkSampler clamp_sampler)
{
    (void)repeat_sampler;
    (void)clamp_sampler;

    std::filesystem::path shader_dir =
        std::filesystem::path(STRINGIFY(VIEWER_DATA_DIR)) /
        "shaders";

    auto shader_path = (shader_dir / "batch_draw.hlsl").string();

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

    StackAlloc tmp_alloc;
    return vk::PipelineShaders(dev, tmp_alloc, shaders,
        Span<const vk::BindingOverride>({}));
}

static void initCommonDrawPipelineInfo(VkPipelineVertexInputStateCreateInfo &vert_info,
                                       VkPipelineInputAssemblyStateCreateInfo &input_assembly_info,
                                       VkPipelineViewportStateCreateInfo &viewport_info,
                                       VkPipelineMultisampleStateCreateInfo &multisample_info,
                                       VkPipelineRasterizationStateCreateInfo &raster_info) 
{
    // Disable auto vertex assembly
    vert_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vert_info.pNext = nullptr;
    vert_info.flags = 0;
    vert_info.vertexBindingDescriptionCount = 0;
    vert_info.pVertexBindingDescriptions = nullptr;
    vert_info.vertexAttributeDescriptionCount = 0;
    vert_info.pVertexAttributeDescriptions = nullptr;

    // Assembly (standard tri indices)
    input_assembly_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly_info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    input_assembly_info.primitiveRestartEnable = VK_FALSE;

    // Viewport (fully dynamic)
    viewport_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_info.viewportCount = 1;
    viewport_info.pViewports = nullptr;
    viewport_info.scissorCount = 1;
    viewport_info.pScissors = nullptr;

    // Multisample
    multisample_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample_info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisample_info.sampleShadingEnable = VK_FALSE;
    multisample_info.alphaToCoverageEnable = VK_FALSE;
    multisample_info.alphaToOneEnable = VK_FALSE;

    // Rasterization
    raster_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster_info.depthClampEnable = VK_FALSE;
    raster_info.rasterizerDiscardEnable = VK_FALSE;
    raster_info.polygonMode = VK_POLYGON_MODE_FILL;
    raster_info.cullMode = VK_CULL_MODE_BACK_BIT;
    raster_info.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster_info.depthBiasEnable = VK_FALSE;
    raster_info.lineWidth = 1.0f;
}

static Pipeline<1> makeDrawPipeline(const vk::Device &dev,
                                    VkPipelineCache pipeline_cache,
                                    VkRenderPass render_pass,
                                    uint32_t num_frames)
{
    auto shaders = makeDrawShaders(dev, VK_NULL_HANDLE, VK_NULL_HANDLE);

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
    blend_attach.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT;

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

#if 0
    VkPushConstantRange push_const {
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        0,
    };
#endif

    // Layout configuration

    std::array<VkDescriptorSetLayout, 3> draw_desc_layouts {{
        shaders.getLayout(0),
        shaders.getLayout(1),
        shaders.getLayout(2)
    }};

    VkPipelineLayoutCreateInfo gfx_layout_info;
    gfx_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    gfx_layout_info.pNext = nullptr;
    gfx_layout_info.flags = 0;
    gfx_layout_info.setLayoutCount =
        static_cast<uint32_t>(draw_desc_layouts.size());
    gfx_layout_info.pSetLayouts = draw_desc_layouts.data();
    gfx_layout_info.pushConstantRangeCount = 0;
    gfx_layout_info.pPushConstantRanges = nullptr;

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

    VkFormat color_format = consts::colorFormat;
    VkFormat depth_format = consts::depthFormat;

    VkPipelineRenderingCreateInfo rendering_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .pNext = nullptr,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &color_format,
        .depthAttachmentFormat = depth_format
    };

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

    vk::FixedDescriptorPool desc_pool(dev, shaders, 0, num_frames);

    return {
        std::move(shaders),
        draw_layout,
        { draw_pipeline },
        std::move(desc_pool),
    };
}

////////////////////////////////////////////////////////////////////////////////
// GENERIC COMPUTE PIPELINE CREATION                                          //
////////////////////////////////////////////////////////////////////////////////

static vk::PipelineShaders makeShaders(const vk::Device &dev,
                                       const char *shader_file,
                                       const char *func_name = "main")
{
    std::filesystem::path shader_dir =
        std::filesystem::path(STRINGIFY(VIEWER_DATA_DIR)) /
        "shaders";

    ShaderCompiler compiler;
    SPIRVShader spirv = compiler.compileHLSLFileToSPV(
        (shader_dir / shader_file).string().c_str(), {},
        {}, {func_name, ShaderStage::Compute });
    
    StackAlloc tmp_alloc;
    return vk::PipelineShaders(dev, tmp_alloc,
                               Span<const SPIRVShader>(&spirv, 1), {});
}

static Pipeline<1> makeComputePipeline(const vk::Device &dev,
                                       VkPipelineCache pipeline_cache,
                                       uint32_t push_constant_size,
                                       uint32_t num_descriptor_sets,
                                       const char *shader_file,
                                       const char *func_name = "main")
{
    vk::PipelineShaders shader = makeShaders(dev, shader_file, func_name);

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

    vk::FixedDescriptorPool desc_pool(dev, shader, 0, num_descriptor_sets);

    return Pipeline<1> {
        std::move(shader),
        layout,
        pipelines,
        std::move(desc_pool),
    };
}

struct BatchFrame {
    BatchImportedBuffers buffers;

    // View, instance info, instance data
    VkDescriptorSet viewInstanceSetPrepare;
    VkDescriptorSet viewInstanceSetDraw;

    HeapArray<LayeredTarget> targets;

    // Contains a copy of the viz buffer output at a certain layer
    vk::LocalImage visualizedCopy;
    VkImageView visualizedCopyView;

    // Contains the processed output from visualizedCopy
    vk::LocalImage visualizedResult;
    VkImageView visualizedResultView;
};

static void makeBatchFrame(vk::Device& dev, 
                           BatchFrame* frame,
                           render::vk::MemoryAllocator &alloc,
                           const BatchRendererProto::Config &cfg,
                           VkDescriptorSet viewInstanceSetPrepare,
                           VkDescriptorSet viewInstanceSetDraw
                           /*vk::FixedDescriptorPool& layerPool*/)
{
    VkDeviceSize view_size = (cfg.numWorlds * cfg.maxViewsPerWorld) * sizeof(PerspectiveCameraData);
    vk::LocalBuffer views = alloc.makeLocalBuffer(view_size).value();

    VkDeviceSize instance_size = (cfg.numWorlds * cfg.maxInstancesPerWorld) * sizeof(InstanceData);
    vk::LocalBuffer instances = alloc.makeLocalBuffer(instance_size).value();

    VkDeviceSize instance_offset_size = (cfg.numWorlds) * sizeof(uint32_t);
    vk::LocalBuffer instance_offsets = alloc.makeLocalBuffer(instance_offset_size).value();

    //Descriptor sets
    std::array<VkWriteDescriptorSet, 6> desc_updates;

    VkDescriptorBufferInfo view_info;
    view_info.buffer = views.buffer;
    view_info.offset = 0;
    view_info.range = view_size;
    vk::DescHelper::storage(desc_updates[0], viewInstanceSetPrepare, &view_info, 0);
    vk::DescHelper::storage(desc_updates[3], viewInstanceSetDraw, &view_info, 0);

    VkDescriptorBufferInfo instance_info;
    instance_info.buffer = instances.buffer;
    instance_info.offset = 0;
    instance_info.range = instance_size;
    vk::DescHelper::storage(desc_updates[1], viewInstanceSetPrepare, &instance_info, 1);
    vk::DescHelper::storage(desc_updates[4], viewInstanceSetDraw, &instance_info, 1);

    VkDescriptorBufferInfo offset_info;
    offset_info.buffer = instance_offsets.buffer;
    offset_info.offset = 0;
    offset_info.range = instance_offset_size;
    vk::DescHelper::storage(desc_updates[2], viewInstanceSetPrepare, &offset_info, 2);
    vk::DescHelper::storage(desc_updates[5], viewInstanceSetDraw, &offset_info, 2);

    vk::LocalImage visualized_copy_img = alloc.makeColorAttachment(cfg.renderWidth, 
                                                              cfg.renderHeight,
                                                              1, consts::outputColorFormat);

    vk::LocalImage visualized_res_img = alloc.makeColorAttachment(cfg.renderWidth, 
                                                              cfg.renderHeight,
                                                              1, VK_FORMAT_R8G8B8A8_UNORM);

    VkImageViewCreateInfo viz_view_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = visualized_copy_img.image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = consts::outputColorFormat,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1
        }
    };

    VkImageView viz_copy_view;
    REQ_VK(dev.dt.createImageView(dev.hdl, &viz_view_info, nullptr, &viz_copy_view));

    viz_view_info.image = visualized_res_img.image;
    viz_view_info.format = VK_FORMAT_R8G8B8A8_UNORM;

    VkImageView viz_res_view;
    REQ_VK(dev.dt.createImageView(dev.hdl, &viz_view_info, nullptr, &viz_res_view));

    vk::DescHelper::update(dev, desc_updates.data(), desc_updates.size());
    new (frame) BatchFrame{
        {std::move(views),std::move(instances),std::move(instance_offsets)},
        viewInstanceSetPrepare,
        viewInstanceSetDraw,
        makeLayeredTargets(cfg.renderWidth, cfg.renderHeight, 
                           cfg.numWorlds * cfg.maxViewsPerWorld,
                           dev, alloc/*, layerPool*/),
        std::move(visualized_copy_img),
        viz_copy_view,
        std::move(visualized_res_img),
        viz_res_view
    };
}

struct ViewBatch {
    // Draw cmds and drawdata
    vk::LocalBuffer drawBuffer;

    // This descriptor set contains draw information
    VkDescriptorSet drawBufferSetPrepare;
    VkDescriptorSet drawBufferSetDraw;

    uint32_t drawCmdOffset;
    uint32_t drawCmdBufferSize;
};

static void makeViewBatch(vk::Device& dev,
                          ViewBatch* batch,
                          render::vk::MemoryAllocator &alloc,
                          VkDescriptorSet draw_buffer_set_prepare,
                          VkDescriptorSet draw_buffer_set_draw)
{
    //Make Draw Buffers

    int64_t buffer_offsets[2];
    int64_t buffer_sizes[3] = {
        (int64_t)sizeof(uint32_t),
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

    vk::DescHelper::storage(desc_updates[0], draw_buffer_set_prepare, &draw_count_info, 0);
    vk::DescHelper::storage(desc_updates[3], draw_buffer_set_draw, &draw_count_info, 0);

    VkDescriptorBufferInfo draw_cmd_info;
    draw_cmd_info.buffer = drawBuffer.buffer;
    draw_cmd_info.offset = buffer_offsets[0];
    draw_cmd_info.range = buffer_sizes[1];

    vk::DescHelper::storage(desc_updates[1], draw_buffer_set_prepare, &draw_cmd_info, 1);
    vk::DescHelper::storage(desc_updates[4], draw_buffer_set_draw, &draw_cmd_info, 1);

    VkDescriptorBufferInfo draw_data_info;
    draw_data_info.buffer = drawBuffer.buffer;
    draw_data_info.offset = buffer_offsets[1];
    draw_data_info.range = buffer_sizes[2];

    vk::DescHelper::storage(desc_updates[2], draw_buffer_set_prepare, &draw_data_info, 2);
    vk::DescHelper::storage(desc_updates[5], draw_buffer_set_draw, &draw_data_info, 2);

    vk::DescHelper::update(dev, desc_updates.data(), desc_updates.size());

    new (batch) ViewBatch{
        std::move(drawBuffer),
        draw_buffer_set_prepare,
        draw_buffer_set_draw,
        (uint32_t)buffer_offsets[0],
        (uint32_t)num_draw_bytes
    };
}

////////////////////////////////////////////////////////////////////////////////
// RASTERIZATION AND RENDERING / POST PROCESSING                              //
////////////////////////////////////////////////////////////////////////////////
static void issueLayoutTransitions(vk::Device &dev, 
                                   Pipeline<1> &draw_pipeline, 
                                   LayeredTarget &target,
                                   VkCommandBuffer &draw_cmd,
                                   ViewBatch &view_batch,
                                   BatchFrame &batch_frame,
                                   VkDescriptorSet asset_set,
                                   VkExtent2D render_extent,
                                   const DynArray<AssetData> &loaded_assets)
{
    // Transition image layouts
    std::array barriers = {
        VkImageMemoryBarrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_NONE,
            .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .image = target.vizBuffer.image,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = target.layerCount
            }
        },
        VkImageMemoryBarrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_NONE,
            .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            .image = target.depth.image,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = target.layerCount
            }
        },
    };

    dev.dt.cmdPipelineBarrier(draw_cmd,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
            0, 0, nullptr, 0, nullptr,
            barriers.size(), barriers.data());   
}

static void prepareVizBlit(vk::Device &dev, 
                           VkCommandBuffer draw_cmd,
                           vk::LocalImage &selected_img,
                           uint32_t layer_id,
                           vk::LocalImage &dst)
{
    // Transition image layouts
    std::array barriers = {
        VkImageMemoryBarrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .image = selected_img.image,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = layer_id,
                .layerCount = 1
            }
        },
        VkImageMemoryBarrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_NONE,
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .image = dst.image,
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
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr,
            barriers.size(), barriers.data());   
}

static void prepareVizCompute(vk::Device &dev, 
                              VkCommandBuffer draw_cmd,
                              vk::LocalImage &selected_img,
                              uint32_t layer_id,
                              vk::LocalImage &dst)
{
    // Transition image layouts
    std::array barriers = {
        VkImageMemoryBarrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .image = selected_img.image,
            .subresourceRange = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = layer_id,
                .layerCount = 1
            }
        },
        VkImageMemoryBarrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_NONE,
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .image = dst.image,
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
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr,
            barriers.size(), barriers.data());   
}

static void issueRasterization(vk::Device &dev, 
                               Pipeline<1> &draw_pipeline, 
                               LayeredTarget &target,
                               VkCommandBuffer &draw_cmd,
                               ViewBatch &view_batch,
                               BatchFrame &batch_frame,
                               VkDescriptorSet asset_set,
                               VkExtent2D render_extent,
                               const DynArray<AssetData> &loaded_assets)
{
    VkRect2D rect = {
        .offset = {},
        .extent = render_extent
    };

    VkRenderingAttachmentInfoKHR color_attach = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR,
        .imageView = target.vizBufferView,
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE
    };

    VkRenderingAttachmentInfoKHR depth_attach = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR,
        .imageView = target.depthView,
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE
    };

    VkRenderingInfo rendering_info = {
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = rect,
        .layerCount = target.layerCount,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attach,
        .pDepthAttachment = &depth_attach
    };

    dev.dt.cmdBeginRenderingKHR(draw_cmd, &rendering_info);

    dev.dt.cmdBindPipeline(draw_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                           draw_pipeline.hdls[0]);

    VkViewport viewport = {
        .width = (float)render_extent.width,
        .height = (float)render_extent.height,
        .minDepth = 0.0f,
        .maxDepth = 1.0f
    };

    dev.dt.cmdSetViewport(draw_cmd, 0, 1, &viewport);
    dev.dt.cmdSetScissor(draw_cmd, 0, 1, &rect);

    dev.dt.cmdBindIndexBuffer(draw_cmd, loaded_assets[0].buf.buffer,
                              loaded_assets[0].idxBufferOffset,
                              VK_INDEX_TYPE_UINT32);

    std::array draw_descriptors = {
        batch_frame.viewInstanceSetDraw,
        view_batch.drawBufferSetDraw,
        asset_set,
    };

    dev.dt.cmdBindDescriptorSets(draw_cmd,
                                 VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 draw_pipeline.layout,
                                 0, 
                                 draw_descriptors.size(),
                                 draw_descriptors.data(), 
                                 0, nullptr);

    dev.dt.cmdDrawIndexedIndirectCount(draw_cmd, 
                                       view_batch.drawBuffer.buffer,
                                       view_batch.drawCmdOffset,
                                       view_batch.drawBuffer.buffer,
                                       0, consts::maxDrawsPerLayeredImage,
                                       sizeof(shader::DrawCmd));

    dev.dt.cmdEndRenderingKHR(draw_cmd);
}

static void issueDeferred(vk::Device &dev,
                          Pipeline<1> &pipeline,
                          LayeredTarget &target,
                          VkCommandBuffer &draw_cmd,
                          ViewBatch &view_batch,
                          BatchFrame &batch_frame,
                          VkDescriptorSet asset_set,
                          VkDescriptorSet asset_mat_tex_set,
                          VkExtent2D render_extent,
                          const DynArray<AssetData> &loaded_assets,
                          uint32_t num_worlds,
                          uint32_t num_instances,
                          uint32_t num_views,
                          uint32_t view_start) {
    { // Transition for compute
        std::array<VkImageMemoryBarrier, 2> compute_prepare {{
            VkImageMemoryBarrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .image = target.vizBuffer.image,
                .subresourceRange = {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = target.layerCount
                }
            },
            VkImageMemoryBarrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .image = target.depth.image,
                .subresourceRange = {
                        .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = target.layerCount
                }
            },
        }};

        dev.dt.cmdPipelineBarrier(draw_cmd,
                                  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  0,
                                  0, nullptr, 0, nullptr,
                                  compute_prepare.size(), compute_prepare.data());
    }

    dev.dt.cmdBindPipeline(draw_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.hdls[0]);

    shader::DeferredLightingPushConstBR push_const = {
            num_views,
            view_start,
            num_views,
            num_instances
    };

    dev.dt.cmdPushConstants(draw_cmd, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(shader::DeferredLightingPushConstBR), &push_const);

    std::array draw_descriptors = {
            target.lightingSet,
            batch_frame.viewInstanceSetDraw,
            asset_set,
            asset_mat_tex_set
    };

    dev.dt.cmdBindDescriptorSets(draw_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 pipeline.layout, 0,
                                 draw_descriptors.size(),
                                 draw_descriptors.data(),
                                 0, nullptr);

    uint32_t num_workgroups_x = target.layerCount;
    uint32_t num_workgroups_y = utils::divideRoundUp(target.vizBuffer.width, 32_u32);
    uint32_t num_workgroups_z = utils::divideRoundUp(target.vizBuffer.height, 32_u32);

    dev.dt.cmdDispatch(draw_cmd, num_workgroups_x, num_workgroups_y, num_workgroups_z);

    { // Transition for compute
        std::array<VkImageMemoryBarrier, 1> compute_prepare {{
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                target.output.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, target.layerCount
                },
            },
        }};

        dev.dt.cmdPipelineBarrier(draw_cmd,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                  0,
                                  0, nullptr, 0, nullptr,
                                  compute_prepare.size(), compute_prepare.data());
    }
}

////////////////////////////////////////////////////////////////////////////////
// BATCH RENDERER PROTOTYPE IMPLEMENTATION                                    //
////////////////////////////////////////////////////////////////////////////////

struct BatchRendererProto::Impl {
    vk::Device &dev;
    vk::MemoryAllocator &mem;
    VkPipelineCache pipelineCache;

    uint32_t maxNumViews;

    // Resources used in/for rendering the batch output
    // We use anything from double, triple, or whatever we can buffering to save
    // on memory usage

    Pipeline<1> prepareViews;
    Pipeline<1> batchDraw;
    Pipeline<1> createVisualization;
    // Pipeline<1> lighting;

    //One batch is num_layers views at once
    HeapArray<ViewBatch> viewBatches;

    //One frame is on simulation frame
    HeapArray<BatchFrame> batchFrames;

    VkDescriptorSet assetSetPrepare;
    VkDescriptorSet assetSetDraw;
    VkDescriptorSet assetSetTextureMat;

    VkExtent2D renderExtent;

    uint32_t selectedView;

    // This pipeline prepares the draw commands in the buffered draw cmds buffer
    // Pipeline<1> prepareViews;

    Impl(const Config &cfg, vk::Device &dev, vk::MemoryAllocator &mem, 
         VkPipelineCache cache, VkDescriptorSet asset_set_comp, 
         VkDescriptorSet asset_set_draw, VkDescriptorSet asset_set_lighting);
};

BatchRendererProto::Impl::Impl(const Config &cfg,
                               vk::Device &dev,
                               vk::MemoryAllocator &mem,
                               VkPipelineCache pipeline_cache, 
                               VkDescriptorSet asset_set_compute,
                               VkDescriptorSet asset_set_draw,
                               VkDescriptorSet asset_set_texture_mat)
    : dev(dev), mem(mem), pipelineCache(pipeline_cache),
      maxNumViews(cfg.numWorlds * cfg.maxViewsPerWorld),
      prepareViews(makeComputePipeline(dev, pipelineCache, 
                                       sizeof(shader::PrepareViewPushConstant),
                                       4+consts::numDrawCmdBuffers,
                                       "prepare_views.hlsl")),
      batchDraw(makeDrawPipeline(dev, pipeline_cache, VK_NULL_HANDLE, consts::numDrawCmdBuffers*cfg.numFrames)),
      createVisualization(makeComputePipeline(dev, pipelineCache,
                                              sizeof(uint32_t) * 2,
                                              cfg.numFrames*consts::numDrawCmdBuffers,
                                              "visualize_tris.hlsl", "visualize")),
      // lighting(makeComputePipeline(dev, pipeline_cache, sizeof(shader::DeferredLightingPushConstBR),
                                   // consts::numDrawCmdBuffers * cfg.numFrames,"draw_deferred.hlsl","lighting")),
      viewBatches(consts::numDrawCmdBuffers),
      batchFrames(cfg.numFrames),
      assetSetPrepare(asset_set_compute),
      assetSetDraw(asset_set_draw),
      assetSetTextureMat(asset_set_texture_mat),
      renderExtent{ cfg.renderWidth, cfg.renderHeight },
      selectedView(0)
{
    printf("Num views total: %d\n", maxNumViews);

    for (uint32_t i = 0; i < consts::numDrawCmdBuffers; i++) {
        makeViewBatch(dev, &viewBatches[i], mem, prepareViews.descPool.makeSet(),
                                                 batchDraw.descPool.makeSet());
    }

    for (uint32_t i = 0; i < cfg.numFrames; i++) {
        makeBatchFrame(dev, &batchFrames[i], mem, cfg,
                       prepareViews.descPool.makeSet(),
                       batchDraw.descPool.makeSet()/*, lighting.descPool*/);
    }
}

BatchRendererProto::BatchRendererProto(const Config &cfg,
                                       vk::Device &dev,
                                       vk::MemoryAllocator &mem,
                                       VkPipelineCache pipeline_cache,
                                       VkDescriptorSet asset_set_compute,
                                       VkDescriptorSet asset_set_draw,
                                       VkDescriptorSet asset_set_texture_mat)
    : impl(std::make_unique<Impl>(cfg, dev, mem, pipeline_cache, 
                                  asset_set_compute, asset_set_draw, asset_set_texture_mat))
{
}

BatchRendererProto::~BatchRendererProto()
{
    impl->dev.dt.destroyPipeline(impl->dev.hdl, impl->prepareViews.hdls[0], nullptr);
    impl->dev.dt.destroyPipelineLayout(impl->dev.hdl, impl->prepareViews.layout, nullptr);
    impl->dev.dt.destroyPipeline(impl->dev.hdl, impl->batchDraw.hdls[0], nullptr);
    impl->dev.dt.destroyPipelineLayout(impl->dev.hdl, impl->batchDraw.layout, nullptr);
    // impl->dev.dt.destroyPipeline(impl->dev.hdl, impl->lighting.hdls[0], nullptr);
    // impl->dev.dt.destroyPipelineLayout(impl->dev.hdl, impl->lighting.layout, nullptr);

    for(int i=0;i<impl->batchFrames.size();i++){
        for(int i2=0;i2<impl->batchFrames[i].targets.size();i2++){
            impl->dev.dt.destroyImageView(impl->dev.hdl, impl->batchFrames[i].targets[i2].vizBufferView, nullptr);
            impl->dev.dt.destroyImageView(impl->dev.hdl, impl->batchFrames[i].targets[i2].depthView, nullptr);
            impl->dev.dt.destroyImageView(impl->dev.hdl, impl->batchFrames[i].targets[i2].outputView, nullptr);
        }
    }
}


static void issuePrepareViewsPipeline(vk::Device& dev,
                                      VkCommandBuffer& draw_cmd,
                                      Pipeline<1>& prepare_views,
                                      BatchFrame& frame,
                                      ViewBatch& batch,
                                      VkDescriptorSet& assetSetPrepareView,
                                      uint32_t num_worlds,
                                      uint32_t num_instances,
                                      uint32_t num_views,
                                      uint32_t view_start,
                                      uint32_t num_processed_batches)
{
    dev.dt.cmdBindPipeline(draw_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           prepare_views.hdls[0]);

    { // Dispatch the compute shader
        std::array view_gen_descriptors = {
            frame.viewInstanceSetPrepare,
            batch.drawBufferSetPrepare,
            assetSetPrepareView,
        };

        dev.dt.cmdBindDescriptorSets(draw_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                     prepare_views.layout, 0,
                                     view_gen_descriptors.size(),
                                     view_gen_descriptors.data(),
                                     0, nullptr);

        shader::PrepareViewPushConstant view_push_const = {
            num_views, view_start, num_worlds, num_instances
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

void BatchRendererProto::renderViews(VkCommandBuffer& draw_cmd, 
                                     BatchRenderInfo info,
                                     const DynArray<AssetData> &loaded_assets) 
{ 
    // Circles between 0 to number of frames
    uint32_t frame_index = 0;

    BatchFrame &frame_data = impl->batchFrames[frame_index];

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
        };

        impl->dev.dt.cmdPipelineBarrier(draw_cmd,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  0, 0, nullptr, barriers.size(), barriers.data(),
                                  0, nullptr);

        for (int i = 0; i < (int)consts::numDrawCmdBuffers; ++i) {
            impl->dev.dt.cmdFillBuffer(draw_cmd, impl->viewBatches[i].drawBuffer.buffer, 
                                       0, sizeof(uint32_t), 0);
        }
    }

    uint32_t num_views = info.numViews;

    struct BatchInfo {
        uint32_t numViews;
        uint32_t offset;
    };

    uint32_t num_batches = utils::divideRoundUp(num_views, impl->dev.maxNumLayersPerImage);
    BatchInfo *batch_infos = (BatchInfo *)alloca(sizeof(BatchInfo) * num_batches);

    { // Populate batch infos
        uint32_t views_left = num_views;
        for (int i = 0; i < (int)num_batches-1; ++i) {
            batch_infos[i].numViews = impl->dev.maxNumLayersPerImage;
            views_left -= impl->dev.maxNumLayersPerImage;
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
                           VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
                           VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        for (int batch = 0; batch < (int)cur_num_batches; ++batch) {
            int batch_no = batch + iter * consts::numDrawCmdBuffers;

            issuePrepareViewsPipeline(impl->dev, draw_cmd, 
                                      impl->prepareViews,
                                      frame_data, 
                                      impl->viewBatches[batch],
                                      impl->assetSetPrepare,
                                      info.numWorlds, 
                                      info.numInstances,
                                      batch_infos[batch_no].numViews, batch_infos[batch_no].offset,
                                      batch_no);
        }

        issueMemoryBarrier(impl->dev, draw_cmd,
                           VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
                           VK_ACCESS_SHADER_READ_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                               VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT |
                               VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

        for (int batch = 0; batch < (int)cur_num_batches; ++batch) {
            int batch_no = batch + iter * consts::numDrawCmdBuffers;
            
            issueLayoutTransitions(impl->dev,
                                   impl->batchDraw,
                                   frame_data.targets[batch_no],
                                   draw_cmd,
                                   impl->viewBatches[batch],
                                   frame_data,
                                   impl->assetSetDraw,
                                   impl->renderExtent,
                                   loaded_assets);
        }

        for (int batch = 0; batch < (int)cur_num_batches; ++batch) {
            int batch_no = batch + iter * consts::numDrawCmdBuffers;

            //Finish rest of draws for the frame
            issueRasterization(impl->dev,
                               impl->batchDraw,
                               frame_data.targets[batch_no],
                               draw_cmd,
                               impl->viewBatches[batch],
                               frame_data,
                               impl->assetSetDraw,
                               impl->renderExtent,
                               loaded_assets);
        }

        issueMemoryBarrier(impl->dev, draw_cmd,
                           VK_ACCESS_SHADER_READ_BIT,
                           VK_ACCESS_TRANSFER_WRITE_BIT,
                           VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                               VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT |
                               VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT);

        for (int batch = 0; batch < (int)cur_num_batches; ++batch) {
            impl->dev.dt.cmdFillBuffer(draw_cmd, impl->viewBatches[batch].drawBuffer.buffer,
                                       0, sizeof(uint32_t), 0);
        }
    }

    issueMemoryBarrier(impl->dev, draw_cmd,
                       VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
                       VK_ACCESS_TRANSFER_READ_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT);

    { // Blit image from mega images to a visualization image
        VkImageBlit blit = {
            .srcSubresource = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = 0,
                .baseArrayLayer = impl->selectedView % impl->dev.maxNumLayersPerImage,
                .layerCount = 1
            },
            .srcOffsets = {
                {}, { (int)impl->renderExtent.width, (int)impl->renderExtent.height, 1 }
            },
            .dstSubresource = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1
            },
            .dstOffsets = {
                {}, { (int)impl->renderExtent.width, (int)impl->renderExtent.height, 1 }
            }
        };

        uint32_t selected_img_index = impl->selectedView / 
                                      impl->dev.maxNumLayersPerImage;
        vk::LocalImage &selected_img = 
            impl->batchFrames[frame_index].targets[selected_img_index].vizBuffer;
        vk::LocalImage &target_dst =
            impl->batchFrames[frame_index].visualizedCopy;

        prepareVizBlit(impl->dev,
                       draw_cmd,
                       selected_img,
                       impl->selectedView % impl->dev.maxNumLayersPerImage,
                       target_dst);

        impl->dev.dt.cmdBlitImage(draw_cmd, 
                                  selected_img.image,
                                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                  impl->batchFrames[frame_index].visualizedCopy.image,
                                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  1, &blit,
                                  VK_FILTER_NEAREST);
    }

    { // Prepare the blitted image and the viz result image for computation
        
    }
}

BatchImportedBuffers &BatchRendererProto::getImportedBuffers(uint32_t frame_id) 
{
    return impl->batchFrames[frame_id].buffers;
}

}
