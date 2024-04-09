#include "render_ctx.hpp"

#include <array>
#include <filesystem>

#include <madrona/render/vk/backend.hpp>
#include <madrona/render/shader_compiler.hpp>
#ifdef MADRONA_VK_CUDA_SUPPORT
#include <madrona/cuda_utils.hpp>
#endif

#include "render_common.hpp"
#include "batch_renderer.hpp"
#include "asset_utils.hpp"
#include "shaders/shader_common.h"
#include "vk/descriptors.hpp"
#include "shader.hpp"

#include <filesystem>
#include <fstream>
#include <signal.h>

#ifdef MADRONA_MACOS
#include <dlfcn.h>
#endif

#include <stb_image.h>

#define ENABLE_BATCH_RENDERER

using std::vector;
using std::min;
using std::max;
using std::array;

namespace madrona::render {

using namespace vk;

using Vertex = render::shader::Vertex;
using PackedVertex = render::shader::PackedVertex;
using MeshData = render::shader::MeshData;
using MaterialData = render::shader::MaterialData;
using ObjectData = render::shader::ObjectData;
using DrawPushConst = render::shader::DrawPushConst;
using CullPushConst = render::shader::CullPushConst;
using DeferredLightingPushConst = render::shader::DeferredLightingPushConst;
using DrawCmd = render::shader::DrawCmd;
using DrawData = render::shader::DrawData;
using PackedInstanceData = render::shader::PackedInstanceData;
using PackedViewData = render::shader::PackedViewData;
using ShadowViewData = render::shader::ShadowViewData;
using DirectionalLight = render::shader::DirectionalLight;
using SkyData = render::shader::SkyData;
using DensityLayer = render::shader::DensityLayer;

void initCommonDrawPipelineInfo(
    VkPipelineVertexInputStateCreateInfo &vert_info,
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

static VkQueue makeGFXQueue(const Device &dev, uint32_t idx)
{
    if (idx >= dev.numGraphicsQueues) {
        FATAL("Not enough graphics queues");
    }

    return makeQueue(dev, dev.gfxQF, idx);
}

static VkPipelineCache getPipelineCache(const Device &dev)
{
    // Pipeline cache (unsaved)
    VkPipelineCacheCreateInfo pcache_info {};
    pcache_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    VkPipelineCache pipeline_cache;
    REQ_VK(dev.dt.createPipelineCache(dev.hdl, &pcache_info, nullptr,
                                      &pipeline_cache));

    return pipeline_cache;
}

static VkRenderPass makeRenderPass(const Device &dev,
                                   VkFormat color_fmt,
                                   VkFormat normal_fmt,
                                   VkFormat position_fmt,
                                   VkFormat depth_fmt)
{
    vector<VkAttachmentDescription> attachment_descs;
    vector<VkAttachmentReference> attachment_refs;

    attachment_descs.push_back(
        {0, color_fmt, VK_SAMPLE_COUNT_1_BIT,
         VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE,
         VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE,
         VK_IMAGE_LAYOUT_UNDEFINED,
         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});

    attachment_descs.push_back(
        {0, normal_fmt, VK_SAMPLE_COUNT_1_BIT,
         VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE,
         VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE,
         VK_IMAGE_LAYOUT_UNDEFINED,
         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});

    attachment_descs.push_back(
        {0, position_fmt, VK_SAMPLE_COUNT_1_BIT,
         VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE,
         VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE,
         VK_IMAGE_LAYOUT_UNDEFINED,
         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});

    attachment_descs.push_back(
        {0, depth_fmt, VK_SAMPLE_COUNT_1_BIT,
         VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_DONT_CARE,
         VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE,
         VK_IMAGE_LAYOUT_UNDEFINED,
         VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL});

    
    attachment_refs.push_back(
        {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
    attachment_refs.push_back(
        {1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
    attachment_refs.push_back(
        {2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
    attachment_refs.push_back(
        {3, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL});

    VkSubpassDescription subpass_desc {};
    subpass_desc.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass_desc.colorAttachmentCount =
        static_cast<uint32_t>(attachment_refs.size() - 1);
    subpass_desc.pColorAttachments = &attachment_refs[0];
    subpass_desc.pDepthStencilAttachment = &attachment_refs.back();

    VkRenderPassCreateInfo render_pass_info;
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_info.pNext = nullptr;
    render_pass_info.flags = 0;
    render_pass_info.attachmentCount =
        static_cast<uint32_t>(attachment_descs.size());
    render_pass_info.pAttachments = attachment_descs.data();
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass_desc;

    render_pass_info.dependencyCount = 0;
    render_pass_info.pDependencies = nullptr;

    VkRenderPass render_pass;
    REQ_VK(dev.dt.createRenderPass(dev.hdl, &render_pass_info, nullptr,
                                   &render_pass));

    return render_pass;
}

static VkRenderPass makeShadowRenderPass(const Device &dev,
                                         VkFormat variance_fmt,
                                         VkFormat depth_fmt)
{
    vector<VkAttachmentDescription> attachment_descs;
    vector<VkAttachmentReference> attachment_refs;

    attachment_descs.push_back(
        {0, variance_fmt, VK_SAMPLE_COUNT_1_BIT,
         VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE,
         VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE,
         VK_IMAGE_LAYOUT_UNDEFINED,
         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});

    attachment_descs.push_back(
        {0, depth_fmt, VK_SAMPLE_COUNT_1_BIT,
         VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE,
         VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE,
         VK_IMAGE_LAYOUT_UNDEFINED,
         VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL});

    attachment_refs.push_back(
        {static_cast<uint32_t>(attachment_refs.size()),
         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});

    attachment_refs.push_back(
        {static_cast<uint32_t>(attachment_refs.size()),
         VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL});

    VkSubpassDescription subpass_desc {};
    subpass_desc.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass_desc.colorAttachmentCount = 1;
    subpass_desc.pColorAttachments = &attachment_refs.front();
    subpass_desc.pDepthStencilAttachment = &attachment_refs.back();

    VkRenderPassCreateInfo render_pass_info;
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_info.pNext = nullptr;
    render_pass_info.flags = 0;
    render_pass_info.attachmentCount =
        static_cast<uint32_t>(attachment_descs.size());
    render_pass_info.pAttachments = attachment_descs.data();
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass_desc;

    render_pass_info.dependencyCount = 0;
    render_pass_info.pDependencies = nullptr;

    VkRenderPass render_pass;
    REQ_VK(dev.dt.createRenderPass(dev.hdl, &render_pass_info, nullptr,
                                   &render_pass));

    return render_pass;
}


static PipelineShaders makeDrawShaders(
    const Device &dev, VkSampler repeat_sampler, VkSampler clamp_sampler)
{
    (void)repeat_sampler;
    (void)clamp_sampler;

    std::filesystem::path shader_dir =
        std::filesystem::path(STRINGIFY(MADRONA_RENDER_DATA_DIR)) /
        "shaders";

    auto shader_path = (shader_dir / "viewer_draw.hlsl").string();

    ShaderCompiler compiler;
    SPIRVShader vert_spirv = compiler.compileHLSLFileToSPV(
        shader_path.c_str(), {}, {},
        { "vert", ShaderStage::Vertex });

    SPIRVShader frag_spirv = compiler.compileHLSLFileToSPV(
        shader_path.c_str(), {}, {},
        { "frag", ShaderStage::Fragment });

#if 0
            {0, 2, repeat_sampler, 1, 0},
            {0, 3, clamp_sampler, 1, 0},
            {1, 1, VK_NULL_HANDLE,
                VulkanConfig::max_materials *
                    VulkanConfig::textures_per_material,
             VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT},
#endif

    std::array<SPIRVShader, 2> shaders {
        std::move(vert_spirv),
        std::move(frag_spirv),
    };

    StackAlloc tmp_alloc;
    return PipelineShaders(dev, tmp_alloc, shaders,
        Span<const BindingOverride>({
            BindingOverride {
                2,
                0,
                VK_NULL_HANDLE,
                InternalConfig::maxTextures,
                VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
            },
            BindingOverride {
                2,
                1,
                repeat_sampler,
                1,
                0,
            },
        }));
}

static PipelineShaders makeCullShader(const Device &dev)
{
    std::filesystem::path shader_dir =
        std::filesystem::path(STRINGIFY(MADRONA_RENDER_DATA_DIR)) /
        "shaders";

    ShaderCompiler compiler;
    SPIRVShader spirv = compiler.compileHLSLFileToSPV(
        (shader_dir / "viewer_cull.hlsl").string().c_str(), {},
        {}, { "instanceCull", ShaderStage::Compute });

    StackAlloc tmp_alloc;
    return PipelineShaders(dev, tmp_alloc,
                           Span<const SPIRVShader>(&spirv, 1), {});
}

static Pipeline<1> makeDrawPipeline(const Device &dev,
                                    VkPipelineCache pipeline_cache,
                                    VkRenderPass render_pass,
                                    VkSampler repeat_sampler,
                                    VkSampler clamp_sampler,
                                    uint32_t num_frames)
{
    auto shaders =
        makeDrawShaders(dev, repeat_sampler, clamp_sampler);

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
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    array<VkPipelineColorBlendAttachmentState, 3> blend_attachments {{
        blend_attach,
        blend_attach,
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
    array dyn_enable {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };

    VkPipelineDynamicStateCreateInfo dyn_info {};
    dyn_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn_info.dynamicStateCount = dyn_enable.size();
    dyn_info.pDynamicStates = dyn_enable.data();

    // Push constant
    VkPushConstantRange push_const {
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        sizeof(DrawPushConst),
    };

    // Layout configuration

    array<VkDescriptorSetLayout, 3> draw_desc_layouts {{
        shaders.getLayout(0),
        shaders.getLayout(1),
        shaders.getLayout(2),
    }};

    VkPipelineLayoutCreateInfo gfx_layout_info;
    gfx_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    gfx_layout_info.pNext = nullptr;
    gfx_layout_info.flags = 0;
    gfx_layout_info.setLayoutCount =
        static_cast<uint32_t>(draw_desc_layouts.size());
    gfx_layout_info.pSetLayouts = draw_desc_layouts.data();
    gfx_layout_info.pushConstantRangeCount = 1;
    gfx_layout_info.pPushConstantRanges = &push_const;

    VkPipelineLayout draw_layout;
    REQ_VK(dev.dt.createPipelineLayout(dev.hdl, &gfx_layout_info, nullptr,
                                       &draw_layout));

    array<VkPipelineShaderStageCreateInfo, 2> gfx_stages {{
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

    VkGraphicsPipelineCreateInfo gfx_info;
    gfx_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gfx_info.pNext = nullptr;
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

    FixedDescriptorPool desc_pool(dev, shaders, 0, num_frames);

    return {
        std::move(shaders),
        draw_layout,
        { draw_pipeline },
        std::move(desc_pool),
    };
}

static Pipeline<1> makeCullPipeline(const Device &dev,
                                    VkPipelineCache pipeline_cache,
                                    CountT num_frames)
{
    PipelineShaders shader = makeCullShader(dev);

    // Push constant
    VkPushConstantRange push_const {
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(CullPushConst),
    };

    // Layout configuration
    std::array desc_layouts {
        shader.getLayout(0),
        shader.getLayout(1),
    };

    VkPipelineLayoutCreateInfo cull_layout_info;
    cull_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    cull_layout_info.pNext = nullptr;
    cull_layout_info.flags = 0;
    cull_layout_info.setLayoutCount =
        static_cast<uint32_t>(desc_layouts.size());
    cull_layout_info.pSetLayouts = desc_layouts.data();
    cull_layout_info.pushConstantRangeCount = 1;
    cull_layout_info.pPushConstantRanges = &push_const;

    VkPipelineLayout cull_layout;
    REQ_VK(dev.dt.createPipelineLayout(dev.hdl, &cull_layout_info, nullptr,
                                       &cull_layout));

    std::array<VkComputePipelineCreateInfo, 1> compute_infos;
#if 0
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT subgroup_size;
    subgroup_size.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT;
    subgroup_size.pNext = nullptr;
    subgroup_size.requiredSubgroupSize = 32;
#endif

    compute_infos[0].sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compute_infos[0].pNext = nullptr;
    compute_infos[0].flags = 0;
    compute_infos[0].stage = {
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        nullptr, //&subgroup_size,
        VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT,
        VK_SHADER_STAGE_COMPUTE_BIT,
        shader.getShader(0),
        "instanceCull",
        nullptr,
    };
    compute_infos[0].layout = cull_layout;
    compute_infos[0].basePipelineHandle = VK_NULL_HANDLE;
    compute_infos[0].basePipelineIndex = -1;

    std::array<VkPipeline, compute_infos.size()> pipelines;
    REQ_VK(dev.dt.createComputePipelines(dev.hdl, pipeline_cache,
                                         compute_infos.size(),
                                         compute_infos.data(), nullptr,
                                         pipelines.data()));

    FixedDescriptorPool desc_pool(dev, shader, 0, num_frames);

    return Pipeline<1> {
        std::move(shader),
        cull_layout,
        pipelines,
        std::move(desc_pool),
    };
}



static EngineInterop setupEngineInterop(Device &dev,
                                        MemoryAllocator &alloc,
                                        bool gpu_input,
                                        uint32_t num_worlds,
                                        uint32_t max_views_per_world,
                                        uint32_t max_instances_per_world,
                                        uint32_t render_width,
                                        uint32_t render_height,
                                        VoxelConfig voxel_config)
{
    (void)dev;

    auto views_cpu = Optional<render::vk::HostBuffer>::none();
    auto view_offsets_cpu = Optional<render::vk::HostBuffer>::none();

    auto instances_cpu = Optional<render::vk::HostBuffer>::none();
    auto instance_offsets_cpu = Optional<render::vk::HostBuffer>::none();
    auto aabb_cpu = Optional<render::vk::HostBuffer>::none();

#ifdef MADRONA_VK_CUDA_SUPPORT
    auto views_gpu = Optional<render::vk::DedicatedBuffer>::none();
    auto views_cuda = Optional<render::vk::CudaImportedBuffer>::none();

    auto view_offsets_gpu = Optional<render::vk::DedicatedBuffer>::none();
    auto view_offsets_cuda = Optional<render::vk::CudaImportedBuffer>::none();
    
    auto instances_gpu = Optional<render::vk::DedicatedBuffer>::none();
    auto instances_cuda = Optional<render::vk::CudaImportedBuffer>::none();

    auto instance_offsets_gpu = Optional<render::vk::DedicatedBuffer>::none();
    auto instance_offsets_cuda = Optional<render::vk::CudaImportedBuffer>::none();

    auto aabb_gpu = Optional<render::vk::DedicatedBuffer>::none();
    auto aabb_cuda = Optional<render::vk::CudaImportedBuffer>::none();
#endif

    VkBuffer views_hdl = VK_NULL_HANDLE;
    VkBuffer view_offsets_hdl = VK_NULL_HANDLE;
    VkBuffer instances_hdl = VK_NULL_HANDLE;
    VkBuffer instance_offsets_hdl = VK_NULL_HANDLE;
    VkBuffer aabb_hdl = VK_NULL_HANDLE;

    void *views_base = nullptr;
    void *view_offsets_base = nullptr;

    void *instances_base = nullptr;
    void *instance_offsets_base = nullptr;

    void *aabb_base = nullptr;

    void *world_ids_instances_base = nullptr;
    void *world_ids_views_base = nullptr;


    { // Create the views buffer
        uint64_t num_views_bytes = num_worlds * max_views_per_world *
            (int64_t)sizeof(render::shader::PackedViewData);

        if (!gpu_input) {
            views_cpu = alloc.makeStagingBuffer(num_views_bytes);
            views_hdl = views_cpu->buffer;

            // views_base = views_cpu->ptr;
            views_base = malloc(sizeof(render::shader::PackedViewData) * num_worlds * max_views_per_world);

            world_ids_instances_base = malloc(sizeof(uint64_t) * num_worlds * max_instances_per_world);
            world_ids_views_base = malloc(sizeof(uint64_t) * num_worlds * max_views_per_world);
        } else {
#ifdef MADRONA_VK_CUDA_SUPPORT
            views_gpu = alloc.makeDedicatedBuffer(
                num_views_bytes, false, true);
            views_cuda.emplace(dev, views_gpu->mem,
                num_views_bytes);

            views_hdl = views_gpu->buf.buffer;
            views_base = (char *)views_cuda->getDevicePointer();
#endif
        }
    }

    { // Create the instances buffer
        uint64_t num_instances_bytes = num_worlds * max_instances_per_world *
            (int64_t)sizeof(render::shader::PackedInstanceData);

        if (!gpu_input) {
            instances_cpu = alloc.makeStagingBuffer(num_instances_bytes);
            instances_hdl = instances_cpu->buffer;
            // instances_base = instances_cpu->ptr;
            instances_base = malloc(sizeof(render::shader::PackedInstanceData) * num_worlds * max_instances_per_world);
        } else {
#ifdef MADRONA_VK_CUDA_SUPPORT
            instances_gpu = alloc.makeDedicatedBuffer(
                num_instances_bytes, false, true);

            instances_cuda.emplace(dev, instances_gpu->mem,
                num_instances_bytes);

            instances_hdl = instances_gpu->buf.buffer;
            instances_base = (char *)instances_cuda->getDevicePointer();
#endif
        }
    }

    { // Create the aabb buffer
        uint64_t num_aabb_bytes = num_worlds * max_instances_per_world *
            (int64_t)sizeof(render::shader::AABB);

        if (!gpu_input) {
            aabb_cpu = alloc.makeStagingBuffer(num_aabb_bytes);
            aabb_hdl = aabb_cpu->buffer;
            // instances_base = instances_cpu->ptr;
            aabb_base = malloc(sizeof(render::shader::AABB) * num_worlds * max_instances_per_world);
        } else {
#ifdef MADRONA_VK_CUDA_SUPPORT
            aabb_gpu = alloc.makeDedicatedBuffer(
                num_aabb_bytes, false, true);
            aabb_cuda.emplace(dev, aabb_gpu->mem,
                num_aabb_bytes);

            aabb_hdl = aabb_gpu->buf.buffer;
            aabb_base = (char *)aabb_cuda->getDevicePointer();
#endif
        }
    }

    { // Create the instance offsets buffer
        uint64_t num_offsets_bytes = (num_worlds+1) * sizeof(int32_t);

        if (!gpu_input) {
            instance_offsets_cpu = alloc.makeStagingBuffer(num_offsets_bytes);
            instance_offsets_hdl = instance_offsets_cpu->buffer;
            instance_offsets_base = instance_offsets_cpu->ptr;
        } else {
#ifdef MADRONA_VK_CUDA_SUPPORT
            instance_offsets_gpu = alloc.makeDedicatedBuffer(
                num_offsets_bytes, false, true);

            instance_offsets_cuda.emplace(dev, instance_offsets_gpu->mem,
                num_offsets_bytes);

            instance_offsets_hdl = instance_offsets_gpu->buf.buffer;
            instance_offsets_base = (char *)instance_offsets_cuda->getDevicePointer();
#endif
        }
    }

    { // Create the instance offsets buffer
        uint64_t num_offsets_bytes = (num_worlds+1) * sizeof(int32_t);

        if (!gpu_input) {
            view_offsets_cpu = alloc.makeStagingBuffer(num_offsets_bytes);
            view_offsets_hdl = view_offsets_cpu->buffer;
            view_offsets_base = view_offsets_cpu->ptr;
        } else {
#ifdef MADRONA_VK_CUDA_SUPPORT
            view_offsets_gpu = alloc.makeDedicatedBuffer(
                num_offsets_bytes, false, true);

            view_offsets_cuda.emplace(dev, view_offsets_gpu->mem,
                num_offsets_bytes);

            view_offsets_hdl = view_offsets_gpu->buf.buffer;
            view_offsets_base = (char *)view_offsets_cuda->getDevicePointer();
#endif
        }
    }

    const uint32_t num_voxels = voxel_config.xLength
        * voxel_config.yLength * voxel_config.zLength;
    const uint32_t staging_size = num_voxels > 0 ? num_voxels * sizeof(int32_t) : 4;

    auto voxel_cpu = Optional<HostBuffer>::none();
    VkBuffer voxel_buffer_hdl = VK_NULL_HANDLE;
    uint32_t *voxel_buffer_ptr = nullptr;

#ifdef MADRONA_VK_CUDA_SUPPORT
    auto voxel_gpu = Optional<render::vk::DedicatedBuffer>::none();
    auto voxel_cuda = Optional<render::vk::CudaImportedBuffer>::none();
#endif

    if (!gpu_input) {
        voxel_cpu = alloc.makeStagingBuffer(staging_size);
        voxel_buffer_ptr = num_voxels ? (uint32_t *)voxel_cpu->ptr : nullptr;
        voxel_buffer_hdl = voxel_cpu->buffer;
    } else {
#ifdef MADRONA_VK_CUDA_SUPPORT
        voxel_gpu = alloc.makeDedicatedBuffer(
            staging_size, false, true);

        voxel_cuda.emplace(
            dev, voxel_gpu->mem, staging_size);

        voxel_buffer_hdl = voxel_gpu->buf.buffer;
        voxel_buffer_ptr = num_voxels ?
            (uint32_t *)voxel_cuda->getDevicePointer() : nullptr;
#endif
    }

    uint32_t *total_num_views_readback = nullptr;
    uint32_t *total_num_instances_readback = nullptr;

    AtomicU32 *total_num_views_cpu_inc = nullptr;
    AtomicU32 *total_num_instances_cpu_inc = nullptr;

    if (!gpu_input) {
        total_num_views_readback = (uint32_t *)malloc(
            2*sizeof(uint32_t));
        total_num_instances_readback = total_num_views_readback + 1;

        total_num_views_cpu_inc = (AtomicU32 *)malloc(2 * sizeof(AtomicU32));
        total_num_instances_cpu_inc = total_num_views_cpu_inc + 1;

        total_num_views_cpu_inc->store_release(0);
        total_num_instances_cpu_inc->store_release(0);
    } else {
#ifdef MADRONA_VK_CUDA_SUPPORT
        total_num_views_readback = (uint32_t *)cu::allocReadback(
            2*sizeof(uint32_t));
        total_num_instances_readback = total_num_views_readback + 1;
#endif
    }

    RenderECSBridge bridge = {
        .views = (PerspectiveCameraData *)views_base,
        .instances = (InstanceData *)instances_base,
        .aabbs = (TLBVHNode *)aabb_base,
        .instanceOffsets = (int32_t *)instance_offsets_base,
        .viewOffsets = (int32_t *)view_offsets_base,
        .totalNumViews = total_num_views_readback,
        .totalNumInstances = total_num_instances_readback,
        .totalNumViewsCPUInc = total_num_views_cpu_inc,
        .totalNumInstancesCPUInc = total_num_instances_cpu_inc,
        .instancesWorldIDs = (uint64_t *)world_ids_instances_base,
        .viewsWorldIDs = (uint64_t *)world_ids_views_base,
        .renderWidth = (int32_t)render_width,
        .renderHeight = (int32_t)render_height,
        .voxels = voxel_buffer_ptr,
        .maxViewsPerworld = max_views_per_world,
        .maxInstancesPerWorld = max_instances_per_world,
        .isGPUBackend = gpu_input
    };

    const RenderECSBridge *gpu_bridge = nullptr;
    if (!gpu_input) {
        gpu_bridge = nullptr;
    } else {
#ifdef MADRONA_VK_CUDA_SUPPORT
        gpu_bridge = (const RenderECSBridge *)cu::allocGPU(
            sizeof(RenderECSBridge));
        cudaMemcpy((void *)gpu_bridge, &bridge, sizeof(RenderECSBridge),
                   cudaMemcpyHostToDevice);
#endif
    }

    uint32_t *iota_array_instances = nullptr;
    uint32_t *iota_array_views = nullptr;
    uint64_t *sorted_instance_world_ids = nullptr;
    uint64_t *sorted_view_world_ids = nullptr;

    if (!gpu_input) {
        iota_array_instances = (uint32_t *)malloc(sizeof(uint32_t) * num_worlds * max_instances_per_world);
        iota_array_views = (uint32_t *)malloc(sizeof(uint32_t) * num_worlds * max_views_per_world);
        sorted_instance_world_ids = (uint64_t *)malloc(sizeof(uint64_t) * num_worlds * max_instances_per_world);
        sorted_view_world_ids = (uint64_t *)malloc(sizeof(uint64_t) * num_worlds * max_views_per_world);
    }

    return EngineInterop {
        std::move(views_cpu),
        std::move(view_offsets_cpu),
        std::move(instances_cpu),
        std::move(instance_offsets_cpu),
        std::move(aabb_cpu),
#ifdef MADRONA_VK_CUDA_SUPPORT
        std::move(views_gpu),
        std::move(view_offsets_gpu),

        std::move(instances_gpu),
        std::move(instance_offsets_gpu),

        std::move(views_cuda),
        std::move(view_offsets_cuda),

        std::move(instances_cuda),
        std::move(instance_offsets_cuda),

        std::move(aabb_gpu),
        std::move(aabb_cuda),
#endif
        views_hdl,
        view_offsets_hdl,
        instances_hdl,
        instance_offsets_hdl,
        aabb_hdl,
        bridge,
        gpu_bridge,
        max_views_per_world,
        max_instances_per_world,
        std::move(voxel_cpu),
#ifdef MADRONA_VK_CUDA_SUPPORT
        std::move(voxel_gpu),
        std::move(voxel_cuda),
#endif
        voxel_buffer_hdl,
        iota_array_instances,
        iota_array_views,
        sorted_instance_world_ids,
        sorted_view_world_ids
    };
}

static size_t fileLength(std::fstream &f)
{
    f.seekg(0, f.end);
    size_t size = f.tellg();
    f.seekg(0, f.beg);

    return size;
}

inline constexpr size_t TRANSMITTANCE_WIDTH = 256;
inline constexpr size_t TRANSMITTANCE_HEIGHT = 64;
inline constexpr size_t SCATTERING_TEXTURE_R_SIZE = 32;
inline constexpr size_t SCATTERING_TEXTURE_MU_SIZE = 128;
inline constexpr size_t SCATTERING_TEXTURE_MU_S_SIZE = 32;
inline constexpr size_t SCATTERING_TEXTURE_NU_SIZE = 8;
inline constexpr size_t SCATTERING_TEXTURE_WIDTH =
    SCATTERING_TEXTURE_NU_SIZE * SCATTERING_TEXTURE_MU_S_SIZE;
inline constexpr size_t SCATTERING_TEXTURE_HEIGHT = SCATTERING_TEXTURE_MU_SIZE;
inline constexpr size_t SCATTERING_TEXTURE_DEPTH = SCATTERING_TEXTURE_R_SIZE;
inline constexpr size_t IRRADIANCE_TEXTURE_WIDTH = 64;
inline constexpr size_t IRRADIANCE_TEXTURE_HEIGHT = 16;
inline constexpr size_t SHADOW_OFFSET_OUTER = 32;
inline constexpr size_t SHADOW_OFFSET_FILTER_SIZE = 8;

static Sky loadSky(const vk::Device &dev, MemoryAllocator &alloc, VkQueue queue)
{
    std::filesystem::path shader_dir =
        std::filesystem::path(STRINGIFY(MADRONA_RENDER_DATA_DIR)) /
        "sky";

    auto [transmittance, transmittance_reqs] = alloc.makeTexture2D(
        TRANSMITTANCE_WIDTH, TRANSMITTANCE_HEIGHT, 1, InternalConfig::skyFormatHighp);
    auto [irradiance, irradiance_reqs] = alloc.makeTexture2D(
        IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT, 1, InternalConfig::skyFormatHighp);
    auto [mie, mie_reqs] = alloc.makeTexture3D(
        SCATTERING_TEXTURE_WIDTH, SCATTERING_TEXTURE_HEIGHT, SCATTERING_TEXTURE_DEPTH, 1, InternalConfig::skyFormatHalfp);
    auto [scattering, scattering_reqs] = alloc.makeTexture3D(
        SCATTERING_TEXTURE_WIDTH, SCATTERING_TEXTURE_HEIGHT, SCATTERING_TEXTURE_DEPTH, 1, InternalConfig::skyFormatHalfp);

    HostBuffer irradiance_hb_staging = alloc.makeStagingBuffer(irradiance_reqs.size);
    HostBuffer mie_hb_staging = alloc.makeStagingBuffer(mie_reqs.size);
    HostBuffer scattering_hb_staging = alloc.makeStagingBuffer(scattering_reqs.size);
    HostBuffer transmittance_hb_staging = alloc.makeStagingBuffer(transmittance_reqs.size);

    std::fstream irradiance_stream(shader_dir / "irradiance.cache", std::fstream::binary | std::fstream::in);
    size_t irradiance_size = fileLength(irradiance_stream);
    irradiance_stream.read((char *)irradiance_hb_staging.ptr, irradiance_size);
    irradiance_hb_staging.flush(dev);

    std::fstream mie_stream(shader_dir / "mie.cache", std::fstream::binary | std::fstream::in);
    size_t mie_size = fileLength(mie_stream);
    mie_stream.read((char *)mie_hb_staging.ptr, mie_size);
    mie_hb_staging.flush(dev);

    std::fstream scattering_stream(shader_dir / "scattering.cache", std::fstream::binary | std::fstream::in);
    size_t scattering_size = fileLength(scattering_stream);
    scattering_stream.read((char *)scattering_hb_staging.ptr, scattering_size);
    scattering_hb_staging.flush(dev);

    std::fstream transmittance_stream(shader_dir / "transmittance.cache", std::fstream::binary | std::fstream::in);
    size_t transmittance_size = fileLength(transmittance_stream);
    transmittance_stream.read((char *)transmittance_hb_staging.ptr, transmittance_size);
    transmittance_hb_staging.flush(dev);

#if 1
    assert(transmittance_size == transmittance_reqs.size && irradiance_size == irradiance_reqs.size &&
           scattering_size == scattering_reqs.size && mie_size == mie_reqs.size);
#endif

    std::optional<VkDeviceMemory> transmittance_backing = alloc.alloc(transmittance_reqs.size);
    std::optional<VkDeviceMemory> irradiance_backing = alloc.alloc(irradiance_reqs.size);
    std::optional<VkDeviceMemory> scattering_backing = alloc.alloc(scattering_reqs.size);
    std::optional<VkDeviceMemory> mie_backing = alloc.alloc(mie_reqs.size);

    assert(transmittance_backing.has_value() && irradiance_backing.has_value() &&
        scattering_backing.has_value() && mie_backing.has_value());

    dev.dt.bindImageMemory(dev.hdl, transmittance.image, transmittance_backing.value(), 0);
    dev.dt.bindImageMemory(dev.hdl, irradiance.image, irradiance_backing.value(), 0);
    dev.dt.bindImageMemory(dev.hdl, scattering.image, scattering_backing.value(), 0);
    dev.dt.bindImageMemory(dev.hdl, mie.image, mie_backing.value(), 0);

    VkCommandPool tmp_pool = makeCmdPool(dev, dev.gfxQF);
    VkCommandBuffer cmdbuf = makeCmdBuffer(dev, tmp_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);

    VkCommandBufferBeginInfo begin_info = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        nullptr,
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        nullptr
    };

    dev.dt.beginCommandBuffer(cmdbuf, &begin_info);
    {
        array<VkImageMemoryBarrier, 4> copy_prepare {{
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                0,
                VK_ACCESS_MEMORY_WRITE_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                transmittance.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            },
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                0,
                VK_ACCESS_MEMORY_WRITE_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                irradiance.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            },
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                0,
                VK_ACCESS_MEMORY_WRITE_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                mie.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            },
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                0,
                VK_ACCESS_MEMORY_WRITE_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                scattering.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            }
        }};

        dev.dt.cmdPipelineBarrier(cmdbuf,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0, nullptr, 0, nullptr,
                copy_prepare.size(), copy_prepare.data());

#if 1
        VkBufferImageCopy copy = {};
        copy.bufferOffset = 0;
        copy.bufferRowLength = 0;
        copy.bufferImageHeight = 0;
        copy.imageExtent.width = TRANSMITTANCE_WIDTH;
        copy.imageExtent.height = TRANSMITTANCE_HEIGHT;
        copy.imageExtent.depth = 1;
        copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.imageSubresource.mipLevel = 0;
        copy.imageSubresource.baseArrayLayer = 0;
        copy.imageSubresource.layerCount = 1;

        dev.dt.cmdCopyBufferToImage(cmdbuf, transmittance_hb_staging.buffer,
            transmittance.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

        copy.imageExtent.width = IRRADIANCE_TEXTURE_WIDTH;
        copy.imageExtent.height = IRRADIANCE_TEXTURE_HEIGHT;
        dev.dt.cmdCopyBufferToImage(cmdbuf, irradiance_hb_staging.buffer,
            irradiance.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

        copy.imageExtent.width = SCATTERING_TEXTURE_WIDTH; 
        copy.imageExtent.height = SCATTERING_TEXTURE_HEIGHT;
        copy.imageExtent.depth = SCATTERING_TEXTURE_DEPTH;
        dev.dt.cmdCopyBufferToImage(cmdbuf, mie_hb_staging.buffer,
            mie.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

        dev.dt.cmdCopyBufferToImage(cmdbuf, scattering_hb_staging.buffer,
            scattering.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

#endif

        array<VkImageMemoryBarrier, 4> finish_prepare {{
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_MEMORY_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                transmittance.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            },
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_MEMORY_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                irradiance.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            },
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_MEMORY_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                mie.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            },
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_MEMORY_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                scattering.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            }
        }};

        dev.dt.cmdPipelineBarrier(cmdbuf,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0, nullptr, 0, nullptr,
                finish_prepare.size(), finish_prepare.data());
    }
    dev.dt.endCommandBuffer(cmdbuf);

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmdbuf;
    submit_info.signalSemaphoreCount = 0;
    submit_info.waitSemaphoreCount = 0;

    dev.dt.queueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);

    dev.dt.deviceWaitIdle(dev.hdl);
    dev.dt.freeCommandBuffers(dev.hdl, tmp_pool, 1, &cmdbuf);
    dev.dt.destroyCommandPool(dev.hdl, tmp_pool, nullptr);

    VkImageViewCreateInfo view_info {};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    VkImageSubresourceRange &view_info_sr = view_info.subresourceRange;
    view_info_sr.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info_sr.baseMipLevel = 0;
    view_info_sr.levelCount = 1;
    view_info_sr.baseArrayLayer = 0;
    view_info_sr.layerCount = 1;

    VkImageView transmittance_view;
    view_info.image = transmittance.image;
    view_info.format = InternalConfig::skyFormatHighp;
    REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &transmittance_view));

    VkImageView irradiance_view;
    view_info.image = irradiance.image;
    view_info.format = InternalConfig::skyFormatHighp;
    REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &irradiance_view));

    VkImageView mie_view;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_3D;
    view_info.image = mie.image;
    view_info.format = InternalConfig::skyFormatHalfp;
    REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &mie_view));

    VkImageView scattering_view;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_3D;
    view_info.image = scattering.image;
    view_info.format = InternalConfig::skyFormatHalfp;
    REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &scattering_view));

    return Sky{
        std::move(transmittance),
        std::move(scattering),
        std::move(mie),
        std::move(irradiance),
        transmittance_view,
        scattering_view,
        mie_view,
        irradiance_view,
        transmittance_backing.value(),
        scattering_backing.value(),
        mie_backing.value(),
        irradiance_backing.value(),
        math::Vector3{ -0.4f, -0.4f, -1.0f },
        math::Vector3{2.0f, 2.0f, 2.0f},
        math::Vector3{0.0046750340586467079f, 0.99998907220740285f, 0.0f},
        20.0f
    };
}

RenderContext::RenderContext(
        APIBackend *render_backend,
        GPUDevice *render_dev,
        const RenderManager::Config &cfg)
    : backend(*static_cast<vk::Backend *>(render_backend)),
      dev(static_cast<vk::Device &>(*render_dev)),
      alloc(dev, backend),
      renderQueue(makeGFXQueue(dev, 0)),
      br_width_(cfg.agentViewWidth),
      br_height_(cfg.agentViewHeight),
      pipelineCache(getPipelineCache(dev)),
      repeatSampler(
          makeImmutableSampler(dev, VK_SAMPLER_ADDRESS_MODE_REPEAT)),
      clampSampler(
          makeImmutableSampler(dev, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)),
      renderPass(makeRenderPass(
          dev, VK_FORMAT_R8G8B8A8_UNORM, InternalConfig::gbufferFormat,
          InternalConfig::gbufferFormat, InternalConfig::depthFormat)),
      shadowPass(makeShadowRenderPass(
          dev, InternalConfig::varianceFormat, InternalConfig::depthFormat)),
      instanceCull(makeCullPipeline(
          dev, pipelineCache, InternalConfig::numFrames)),
      objectDraw(makeDrawPipeline(dev, pipelineCache,
          renderPass, repeatSampler, clampSampler,
          InternalConfig::numFrames)),
      asset_desc_pool_cull_(dev, instanceCull.shaders, 1, 1),
      asset_desc_pool_draw_(dev, objectDraw.shaders, 1, 1),
      asset_desc_pool_mat_tx_(dev, objectDraw.shaders, 2, 1),
      asset_set_cull_(asset_desc_pool_cull_.makeSet()),
      asset_set_draw_(asset_desc_pool_draw_.makeSet()),
      asset_set_mat_tex_(asset_desc_pool_mat_tx_.makeSet()),
      load_cmd_pool_(makeCmdPool(dev, dev.gfxQF)),
      load_cmd_(makeCmdBuffer(dev, load_cmd_pool_)),
      load_fence_(makeFence(dev)),
      engine_interop_(setupEngineInterop(
          dev, alloc, cfg.execMode == ExecMode::CUDA, cfg.numWorlds,
          cfg.maxViewsPerWorld, cfg.maxInstancesPerWorld,
          br_width_, br_height_, cfg.voxelCfg)),
      lights_(InternalConfig::maxLights),
      loaded_assets_(0),
      sky_(loadSky(dev, alloc, renderQueue)),
      material_textures_(0),
      voxel_config_(cfg.voxelCfg),
      num_worlds_(cfg.numWorlds),
      gpu_input_(cfg.execMode == ExecMode::CUDA)
{
    {
        VkDescriptorPoolSize pool_sizes[] = {
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 20 },
            { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, InternalConfig::maxTextures*2 },
            { VK_DESCRIPTOR_TYPE_SAMPLER, 1 }
        };

        VkDescriptorPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pool_info.maxSets = 10 + InternalConfig::maxTextures + 1;
        pool_info.poolSizeCount = 3;
        pool_info.pPoolSizes = pool_sizes;
        REQ_VK(dev.dt.createDescriptorPool(dev.hdl,
            &pool_info, nullptr, &asset_pool_));
    }

    {
        VkDescriptorSetLayoutBinding binding = {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
        };

        VkDescriptorSetLayoutCreateInfo info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 1,
            .pBindings = &binding
        };

        dev.dt.createDescriptorSetLayout(dev.hdl, &info, nullptr, &asset_layout_);
    }

    {
        VkDescriptorSetLayoutBinding bindings[] = {
            {
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
            },
            {
                .binding = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
            },
            {
                .binding = 2,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
            }
        };

        VkDescriptorSetLayoutCreateInfo info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 3,
            .pBindings = bindings
        };

        dev.dt.createDescriptorSetLayout(dev.hdl, &info, nullptr, &asset_batch_lighting_layout_);
    }

    {
        VkDescriptorBindingFlags flags[] = { VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT, 0 };

        VkDescriptorSetLayoutBindingFlagsCreateInfo flag_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
            .bindingCount = 2,
            .pBindingFlags = flags
        };

        VkDescriptorSetLayoutBinding bindings[] = {
            {
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                .descriptorCount = InternalConfig::maxTextures,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
            },

            {
                .binding = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = &repeatSampler
            }
        };

        VkDescriptorSetLayoutCreateInfo info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = &flag_info,
            .bindingCount = 2,
            .pBindings = bindings
        };

        dev.dt.createDescriptorSetLayout(dev.hdl, &info, nullptr, &asset_tex_layout_);
    }

    {
        VkDescriptorSetAllocateInfo alloc_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = asset_pool_,
            .descriptorSetCount = 1,
            .pSetLayouts = &asset_tex_layout_
        };

        dev.dt.allocateDescriptorSets(dev.hdl, &alloc_info, &asset_set_tex_compute_);

        alloc_info.pSetLayouts = &asset_batch_lighting_layout_;
        dev.dt.allocateDescriptorSets(dev.hdl, &alloc_info, &asset_batch_lighting_set_);
    }

#ifdef MADRONA_MACOS
    // Batch renderer is not supported on MacOS
    assert(!cfg.enableBatchRenderer);
#endif

    BatchRenderer::Config br_cfg = {
         cfg.enableBatchRenderer,
         br_width_,
         br_height_,
         cfg.numWorlds,
         cfg.maxViewsPerWorld,
         cfg.maxInstancesPerWorld,
         1
    };

    batchRenderer = std::make_unique<BatchRenderer>(br_cfg, *this);
}

RenderContext::~RenderContext()
{
    waitForIdle();
    
    loaded_assets_.clear();

    for (auto &tx : material_textures_) {
        dev.dt.destroyImageView(dev.hdl, tx.view, nullptr);
        dev.dt.destroyImage(dev.hdl, tx.image.image, nullptr);
        dev.dt.freeMemory(dev.hdl, tx.backing, nullptr);
    }

    dev.dt.destroyImageView(dev.hdl, sky_.transmittanceView, nullptr);
    dev.dt.destroyImageView(dev.hdl, sky_.irradianceView, nullptr);
    dev.dt.destroyImageView(dev.hdl, sky_.mieView, nullptr);
    dev.dt.destroyImageView(dev.hdl, sky_.scatteringView, nullptr);

    dev.dt.freeMemory(dev.hdl, sky_.transmittanceBacking, nullptr);
    dev.dt.freeMemory(dev.hdl, sky_.irradianceBacking, nullptr);
    dev.dt.freeMemory(dev.hdl, sky_.mieBacking, nullptr);
    dev.dt.freeMemory(dev.hdl, sky_.scatteringBacking, nullptr);

    dev.dt.destroyImage(dev.hdl, sky_.transmittance.image, nullptr);
    dev.dt.destroyImage(dev.hdl, sky_.irradiance.image, nullptr);
    dev.dt.destroyImage(dev.hdl, sky_.singleMieScattering.image, nullptr);
    dev.dt.destroyImage(dev.hdl, sky_.scattering.image, nullptr);

    dev.dt.destroyFence(dev.hdl, load_fence_, nullptr);
    dev.dt.destroyCommandPool(dev.hdl, load_cmd_pool_, nullptr);

    dev.dt.destroyDescriptorSetLayout(dev.hdl, asset_layout_, nullptr);
    dev.dt.destroyDescriptorSetLayout(dev.hdl, asset_tex_layout_, nullptr);
    dev.dt.destroyDescriptorSetLayout(dev.hdl, asset_batch_lighting_layout_, nullptr);
    // dev.dt.destroyDescriptorSetLayout(dev.hdl, sky_data_layout_, nullptr);

    dev.dt.destroyDescriptorPool(dev.hdl, asset_pool_, nullptr);

    dev.dt.destroyPipeline(dev.hdl, objectDraw.hdls[0], nullptr);
    dev.dt.destroyPipelineLayout(dev.hdl, objectDraw.layout, nullptr);

    dev.dt.destroyPipeline(dev.hdl, instanceCull.hdls[0], nullptr);
    dev.dt.destroyPipelineLayout(dev.hdl, instanceCull.layout, nullptr);

    dev.dt.destroyRenderPass(dev.hdl, renderPass, nullptr);
    dev.dt.destroyRenderPass(dev.hdl, shadowPass, nullptr);

    dev.dt.destroySampler(dev.hdl, clampSampler, nullptr);
    dev.dt.destroySampler(dev.hdl, repeatSampler, nullptr);

    dev.dt.destroyPipelineCache(dev.hdl, pipelineCache, nullptr);
}

static DynArray<MaterialTexture> loadTextures(
    const vk::Device &dev, MemoryAllocator &alloc, VkQueue queue,
    Span<const imp::SourceTexture> textures)
{
    DynArray<HostBuffer> host_buffers(0);
    DynArray<MaterialTexture> dst_textures(0);

    VkCommandPool tmp_pool = makeCmdPool(dev, dev.gfxQF);

    VkCommandBuffer cmdbuf = makeCmdBuffer(dev, tmp_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);

    VkCommandBufferBeginInfo begin_info = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        nullptr,
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        nullptr
    };

    dev.dt.beginCommandBuffer(cmdbuf, &begin_info);

    for (const imp::SourceTexture &tx : textures)
    {
        const char *filename = tx.path;
        int width, height, components;
        void *pixels = stbi_load(filename, &width, &height, &components, STBI_rgb_alpha);

        auto [texture, texture_reqs] = alloc.makeTexture2D(
                width, height, 1, VK_FORMAT_R8G8B8A8_SRGB);

        HostBuffer texture_hb_staging = alloc.makeStagingBuffer(texture_reqs.size);
        memcpy(texture_hb_staging.ptr, pixels, width * height * 4 * sizeof(char));
        texture_hb_staging.flush(dev);
        stbi_image_free(pixels);

        std::optional<VkDeviceMemory> texture_backing = alloc.alloc(texture_reqs.size);

        assert(texture_backing.has_value());

        dev.dt.bindImageMemory(dev.hdl, texture.image, texture_backing.value(), 0);

        VkImageMemoryBarrier copy_prepare {
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            nullptr,
            0,
            VK_ACCESS_MEMORY_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            texture.image,
            {
                VK_IMAGE_ASPECT_COLOR_BIT,
                0, 1, 0, 1
            },
        };

        dev.dt.cmdPipelineBarrier(cmdbuf,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0, nullptr, 0, nullptr,
            1, &copy_prepare);

        VkBufferImageCopy copy = {};
        copy.bufferOffset = 0;
        copy.bufferRowLength = 0;
        copy.bufferImageHeight = 0;
        copy.imageExtent.width = width;
        copy.imageExtent.height = height;
        copy.imageExtent.depth = 1;
        copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.imageSubresource.mipLevel = 0;
        copy.imageSubresource.baseArrayLayer = 0;
        copy.imageSubresource.layerCount = 1;

        dev.dt.cmdCopyBufferToImage(cmdbuf, texture_hb_staging.buffer,
            texture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

        VkImageMemoryBarrier finish_prepare {
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_MEMORY_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            texture.image,
            {
                VK_IMAGE_ASPECT_COLOR_BIT,
                0, 1, 0, 1
            },
        };
        

        dev.dt.cmdPipelineBarrier(cmdbuf,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0, nullptr, 0, nullptr,
            1, &finish_prepare);

        VkImageViewCreateInfo view_info {};
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        VkImageSubresourceRange &view_info_sr = view_info.subresourceRange;
        view_info_sr.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view_info_sr.baseMipLevel = 0;
        view_info_sr.levelCount = 1;
        view_info_sr.baseArrayLayer = 0;
        view_info_sr.layerCount = 1;

        VkImageView view;
        view_info.image = texture.image;
        view_info.format = VK_FORMAT_R8G8B8A8_SRGB;
        REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &view));

        host_buffers.push_back(std::move(texture_hb_staging));

        dst_textures.emplace_back(std::move(texture), view, texture_backing.value());
    }

    dev.dt.endCommandBuffer(cmdbuf);

    VkSubmitInfo submit_info = {};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmdbuf;
    submit_info.signalSemaphoreCount = 0;
    submit_info.waitSemaphoreCount = 0;

    dev.dt.queueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);

    dev.dt.deviceWaitIdle(dev.hdl);
    dev.dt.freeCommandBuffers(dev.hdl, tmp_pool, 1, &cmdbuf);
    dev.dt.destroyCommandPool(dev.hdl, tmp_pool, nullptr);

    return dst_textures;
}

CountT RenderContext::loadObjects(Span<const imp::SourceObject> src_objs,
                                        Span<const imp::SourceMaterial> src_mats,
                                        Span<const imp::SourceTexture> textures)
{
    using namespace imp;
    using namespace math;

    assert(loaded_assets_.size() == 0);

    int64_t num_total_vertices = 0;
    int64_t num_total_indices = 0;
    int64_t num_total_meshes = 0;

    for (const SourceObject &obj : src_objs) {
        num_total_meshes += obj.meshes.size();

        for (const SourceMesh &mesh : obj.meshes) {
            if (mesh.faceCounts != nullptr) {
                FATAL("Render mesh isn't triangular");
            }

            num_total_vertices += mesh.numVertices;
            num_total_indices += mesh.numFaces * 3;
        }
    }

    int64_t num_total_objs = src_objs.size();

    int64_t buffer_offsets[4];
    int64_t buffer_sizes[5] = {
        (int64_t)sizeof(ObjectData) * num_total_objs,
        (int64_t)sizeof(MeshData) * num_total_meshes,
        (int64_t)sizeof(PackedVertex) * num_total_vertices,
        (int64_t)sizeof(uint32_t) * num_total_indices,
        (int64_t)sizeof(MaterialData) * src_mats.size()
    };
    int64_t num_asset_bytes = utils::computeBufferOffsets(
        buffer_sizes, buffer_offsets, 256);

    HostBuffer staging = alloc.makeStagingBuffer(num_asset_bytes);
    char *staging_ptr = (char *)staging.ptr;
    ObjectData *obj_ptr = (ObjectData *)staging_ptr;
    MeshData *mesh_ptr = 
        (MeshData *)(staging_ptr + buffer_offsets[0]);
    PackedVertex *vertex_ptr =
        (PackedVertex *)(staging_ptr + buffer_offsets[1]);
    uint32_t *indices_ptr =
        (uint32_t *)(staging_ptr + buffer_offsets[2]);
    MaterialData *materials_ptr =
        (MaterialData *)(staging_ptr + buffer_offsets[3]);

    int32_t mesh_offset = 0;
    int32_t vertex_offset = 0;
    int32_t index_offset = 0;
    for (const SourceObject &obj : src_objs) {
        *obj_ptr++ = ObjectData {
            .meshOffset = mesh_offset,
            .numMeshes = (int32_t)obj.meshes.size(),
        };

        for (const SourceMesh &mesh : obj.meshes) {
            int32_t num_mesh_verts = (int32_t)mesh.numVertices;
            int32_t num_mesh_indices = (int32_t)mesh.numFaces * 3;

            mesh_ptr[mesh_offset++] = MeshData {
                .vertexOffset = vertex_offset,
                .numVertices = num_mesh_verts,
                .indexOffset = index_offset,
                .numIndices = num_mesh_indices,
                .materialIndex = (int32_t)mesh.materialIDX
            };

            // Compute new normals
            auto new_normals = Optional<HeapArray<Vector3>>::none();
            if (!mesh.normals) {
                new_normals.emplace(num_mesh_verts);

                for (int64_t vert_idx = 0; vert_idx < num_mesh_verts;
                     vert_idx++) {
                    (*new_normals)[vert_idx] = Vector3::zero();
                }

                for (CountT face_idx = 0; face_idx < (CountT)mesh.numFaces;
                     face_idx++) {
                    CountT base_idx = face_idx * 3;
                    uint32_t i0 = mesh.indices[base_idx];
                    uint32_t i1 = mesh.indices[base_idx + 1];
                    uint32_t i2 = mesh.indices[base_idx + 2];

                    Vector3 v0 = mesh.positions[i0];
                    Vector3 v1 = mesh.positions[i1];
                    Vector3 v2 = mesh.positions[i2];

                    Vector3 e0 = v1 - v0;
                    Vector3 e1 = v2 - v0;

                    Vector3 face_normal = cross(e0, e1);
                    float face_len = face_normal.length();
                    assert(face_len != 0);
                    face_normal /= face_len;

                    (*new_normals)[i0] += face_normal;
                    (*new_normals)[i1] += face_normal;
                    (*new_normals)[i2] += face_normal;
                }

                for (int64_t vert_idx = 0; vert_idx < num_mesh_verts;
                     vert_idx++) {
                    (*new_normals)[vert_idx] =
                        normalize((*new_normals)[vert_idx]);
                }
            }

            for (int32_t i = 0; i < num_mesh_verts; i++) {
                Vector3 pos = mesh.positions[i];
                Vector3 normal = mesh.normals ?
                    mesh.normals[i] : (*new_normals)[i];
                Vector4 tangent_sign;
                // FIXME: use mikktspace at import time
                if (mesh.tangentAndSigns != nullptr) {
                    tangent_sign = mesh.tangentAndSigns[i];
                } else {
                    Vector3 a, b;
                    normal.frame(&a, &b);
                    tangent_sign = {
                        a.x,
                        a.y,
                        a.z,
                        1.f,
                    };
                }
                Vector2 uv = mesh.uvs ? mesh.uvs[i] : Vector2 { 0, 0 };

                Vector3 encoded_normal_tangent =
                    encodeNormalTangent(normal, tangent_sign);

                vertex_ptr[vertex_offset++] = PackedVertex {
                    Vector4 {
                        pos.x,
                        pos.y,
                        pos.z,
                        encoded_normal_tangent.x,
                    },
                    Vector4 {
                        encoded_normal_tangent.y,
                        encoded_normal_tangent.z,
                        uv.x,
                        uv.y,
                    },
                };
            }

            memcpy(indices_ptr + index_offset,
                   mesh.indices, sizeof(uint32_t) * num_mesh_indices);

            index_offset += num_mesh_indices;
        }
    }

    uint32_t mat_idx = 0;
    for (const SourceMaterial &mat : src_mats) {
        materials_ptr[mat_idx].color = mat.color;
        materials_ptr[mat_idx].roughness = mat.roughness;
        materials_ptr[mat_idx].metalness = mat.metalness;
        materials_ptr[mat_idx++].textureIdx = mat.textureIdx;
    }

    staging.flush(dev);

    LocalBuffer asset_buffer = *alloc.makeLocalBuffer(num_asset_bytes);
    GPURunUtil gpu_run {
        load_cmd_pool_,
        load_cmd_,
        renderQueue, // FIXME
        load_fence_
    };

    gpu_run.begin(dev);

    VkBufferCopy buffer_copy {
        .srcOffset = 0,
        .dstOffset = 0,
        .size = (VkDeviceSize)num_asset_bytes,
    };

    dev.dt.cmdCopyBuffer(load_cmd_, staging.buffer, asset_buffer.buffer,
                         1, &buffer_copy);
    

    gpu_run.submit(dev);

    VkDescriptorSet index_buffer_set;
    {
        VkDescriptorSetAllocateInfo alloc_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = asset_pool_,
            .descriptorSetCount = 1,
            .pSetLayouts = &asset_layout_
        };

        dev.dt.allocateDescriptorSets(dev.hdl, &alloc_info, &index_buffer_set);
    }

    DynArray<VkWriteDescriptorSet> desc_updates(8 + (material_textures_.size() > 0 ? 2 : 0));

    VkDescriptorBufferInfo obj_info;
    obj_info.buffer = asset_buffer.buffer;
    obj_info.offset = 0;
    obj_info.range = buffer_sizes[0];

    desc_updates.push_back({});
    DescHelper::storage(desc_updates[0], asset_set_cull_, &obj_info, 0);

    VkDescriptorBufferInfo mesh_info;
    mesh_info.buffer = asset_buffer.buffer;
    mesh_info.offset = buffer_offsets[0];
    mesh_info.range = buffer_sizes[1];

    desc_updates.push_back({});
    DescHelper::storage(desc_updates[1], asset_set_cull_, &mesh_info, 1);

    VkDescriptorBufferInfo vert_info;
    vert_info.buffer = asset_buffer.buffer;
    vert_info.offset = buffer_offsets[1];
    vert_info.range = buffer_sizes[2];

    desc_updates.push_back({});
    DescHelper::storage(desc_updates[2], asset_set_draw_, &vert_info, 0);

    VkDescriptorBufferInfo mat_info;
    mat_info.buffer = asset_buffer.buffer;
    mat_info.offset = buffer_offsets[3];
    mat_info.range = buffer_sizes[4];

    desc_updates.push_back({});
    DescHelper::storage(desc_updates[3], asset_set_draw_, &mat_info, 1);

    VkDescriptorBufferInfo index_set_info;
    index_set_info.buffer = asset_buffer.buffer;
    index_set_info.offset = buffer_offsets[2];
    index_set_info.range = buffer_offsets[3] - buffer_offsets[2];

    desc_updates.push_back({});
    DescHelper::storage(desc_updates[4], index_buffer_set, &index_set_info, 0);

    desc_updates.push_back({});
    DescHelper::storage(desc_updates[5], asset_batch_lighting_set_, &vert_info, 0);

    desc_updates.push_back({});
    DescHelper::storage(desc_updates[6], asset_batch_lighting_set_, &mesh_info, 1);

    desc_updates.push_back({});
    DescHelper::storage(desc_updates[7], asset_batch_lighting_set_, &mat_info, 2);

    material_textures_ = loadTextures(dev, alloc, renderQueue, textures);

    DynArray<VkDescriptorImageInfo> tx_infos(material_textures_.size()+1);
    for (auto &tx : material_textures_) {
        tx_infos.push_back({
                VK_NULL_HANDLE,
                tx.view,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                });
    }

    if (material_textures_.size()) {
        desc_updates.push_back({});
        DescHelper::textures(desc_updates[8], asset_set_mat_tex_, tx_infos.data(), tx_infos.size(), 0);

        desc_updates.push_back({});
        DescHelper::textures(desc_updates[9], asset_set_tex_compute_, tx_infos.data(), tx_infos.size(), 0);
    }

    DescHelper::update(dev, desc_updates.data(), desc_updates.size());

    AssetData asset_data {
        std::move(asset_buffer),
        (uint32_t)buffer_offsets[2],
        index_buffer_set
    };

    loaded_assets_.emplace_back(std::move(asset_data));

    return 0;
}

void RenderContext::configureLighting(Span<const LightConfig> lights)
{
    for (int i = 0; i < lights.size(); ++i) {
        lights_.insert(i, DirectionalLight{ 
            math::Vector4{lights[i].dir.x, lights[i].dir.y, lights[i].dir.z, 1.0f }, 
            math::Vector4{lights[i].color.x, lights[i].color.y, lights[i].color.z, 1.0f}
        });
    }
}

void RenderContext::waitForIdle()
{
    dev.dt.deviceWaitIdle(dev.hdl);
}

}
