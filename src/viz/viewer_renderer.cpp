#include "viewer_renderer.hpp"

#include <filesystem>

#include "backends/imgui_impl_vulkan.h"
#include "backends/imgui_impl_glfw.h"

#include "render_common.hpp"
#include "render_ctx.hpp"

#include <stb_image_write.h>

namespace madrona::viz {

struct ImGUIVkLookupData {
    PFN_vkGetDeviceProcAddr getDevAddr;
    VkDevice dev;
    PFN_vkGetInstanceProcAddr getInstAddr;
    VkInstance inst;
};

using namespace render;
using namespace render::vk;
using std::array;

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

static void imguiVkCheck(VkResult res)
{
    REQ_VK(res);
}

static VkRenderPass makeImGuiRenderPass(const Device &dev,
                                        VkFormat color_fmt,
                                        VkFormat depth_fmt)
{
    array<VkAttachmentDescription, 2> attachment_descs {{
        {
            0,
            color_fmt,
            VK_SAMPLE_COUNT_1_BIT,
            VK_ATTACHMENT_LOAD_OP_LOAD,
            VK_ATTACHMENT_STORE_OP_STORE,
            VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            VK_ATTACHMENT_STORE_OP_DONT_CARE,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        },
        {
            0,
            depth_fmt,
            VK_SAMPLE_COUNT_1_BIT,
            VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            VK_ATTACHMENT_STORE_OP_DONT_CARE,
            VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            VK_ATTACHMENT_STORE_OP_DONT_CARE,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        },
    }};

    array<VkAttachmentReference, 2> attachment_refs {{
        {
            0,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        },
        {
            1,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        },
    }};
    
    VkSubpassDescription subpass_desc {};
    subpass_desc.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass_desc.colorAttachmentCount = attachment_refs.size() - 1;
    subpass_desc.pColorAttachments = attachment_refs.data();
    subpass_desc.pDepthStencilAttachment = &attachment_refs.back();

    VkRenderPassCreateInfo render_pass_info;
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_info.pNext = nullptr;
    render_pass_info.flags = 0;
    render_pass_info.attachmentCount = attachment_descs.size();
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

static PFN_vkVoidFunction imguiVKLookup(const char *fname,
                                        void *user_data)
{
    auto data = (ImGUIVkLookupData *)user_data;

    auto addr = data->getDevAddr(data->dev, fname);

    if (!addr) {
        addr = data->getInstAddr(data->inst, fname);
    }

    if (!addr) {
        FATAL("Failed to load ImGUI vulkan function: %s", fname);
    }

    return addr;
}

static ImGuiRenderState imguiInit(GLFWwindow *window, const Device &dev,
                                  const Backend &backend, VkQueue ui_queue,
                                  VkPipelineCache pipeline_cache,
                                  VkFormat color_fmt,
                                  VkFormat depth_fmt)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    auto font_path = std::filesystem::path(STRINGIFY(MADRONA_RENDER_DATA_DIR)) /
        "font.ttf";

    float scale_factor;
    {
        float x_scale, y_scale;
        glfwGetWindowContentScale(window, &x_scale, &y_scale);
        assert(x_scale == y_scale);

        scale_factor = x_scale;
    }

    float scaled_font_size = 16.f * scale_factor;
    io.Fonts->AddFontFromFileTTF(font_path.string().c_str(), scaled_font_size);

    auto &style = ImGui::GetStyle();
    style.ScaleAllSizes(scale_factor);

    ImGui_ImplGlfw_InitForVulkan(window, true);

    // Taken from imgui/examples/example_glfw_vulkan/main.cpp
    VkDescriptorPool desc_pool;
    {
        VkDescriptorPoolSize pool_sizes[] = {
            { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
            { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
            { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 },
        };
        VkDescriptorPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pool_info.maxSets = 1000 * IM_ARRAYSIZE(pool_sizes);
        pool_info.poolSizeCount = (uint32_t)IM_ARRAYSIZE(pool_sizes);
        pool_info.pPoolSizes = pool_sizes;
        REQ_VK(dev.dt.createDescriptorPool(dev.hdl,
            &pool_info, nullptr, &desc_pool));
    }

    ImGui_ImplVulkan_InitInfo vk_init = {};
    vk_init.Instance = backend.hdl;
    vk_init.PhysicalDevice = dev.phy;
    vk_init.Device = dev.hdl;
    vk_init.QueueFamily = dev.gfxQF;
    vk_init.Queue = ui_queue;
    vk_init.PipelineCache = pipeline_cache;
    vk_init.DescriptorPool = desc_pool;
    vk_init.MinImageCount = InternalConfig::numFrames;
    vk_init.ImageCount = InternalConfig::numFrames;
    vk_init.CheckVkResultFn = imguiVkCheck;

    VkRenderPass imgui_renderpass = makeImGuiRenderPass(dev, color_fmt,
                                                        depth_fmt);

    ImGUIVkLookupData lookup_data {
        dev.dt.getDeviceProcAddr,
        dev.hdl,
        backend.dt.getInstanceProcAddr,
        backend.hdl,
    };
    ImGui_ImplVulkan_LoadFunctions(imguiVKLookup, &lookup_data);
    ImGui_ImplVulkan_Init(&vk_init, imgui_renderpass);

    VkCommandPool tmp_pool = makeCmdPool(dev, dev.gfxQF);
    VkCommandBuffer tmp_cmd = makeCmdBuffer(dev, tmp_pool);
    VkCommandBufferBeginInfo tmp_begin {};
    tmp_begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    REQ_VK(dev.dt.beginCommandBuffer(tmp_cmd, &tmp_begin));
    ImGui_ImplVulkan_CreateFontsTexture(tmp_cmd);
    REQ_VK(dev.dt.endCommandBuffer(tmp_cmd));

    VkFence tmp_fence = makeFence(dev);

    VkSubmitInfo tmp_submit {
        VK_STRUCTURE_TYPE_SUBMIT_INFO,
        nullptr,
        0,
        nullptr,
        nullptr,
        1,
        &tmp_cmd,
        0,
        nullptr,
    };

    REQ_VK(dev.dt.queueSubmit(ui_queue, 1, &tmp_submit, tmp_fence));
    waitForFenceInfinitely(dev, tmp_fence);

    dev.dt.destroyFence(dev.hdl, tmp_fence, nullptr);
    dev.dt.destroyCommandPool(dev.hdl, tmp_pool, nullptr);

    return {
        desc_pool,
        imgui_renderpass,
    };
}

static PipelineShaders makeVoxelGenShader(const Device& dev)
{
    std::filesystem::path shader_dir =
        std::filesystem::path(STRINGIFY(MADRONA_RENDER_DATA_DIR)) /
        "shaders";

    ShaderCompiler compiler;
    SPIRVShader spirv = compiler.compileHLSLFileToSPV(
        (shader_dir / "voxel_gen.hlsl").string().c_str(), {},
        {}, { "voxelGen", ShaderStage::Compute });

    StackAlloc tmp_alloc;
    return PipelineShaders(
        dev, tmp_alloc,
        Span<const SPIRVShader>(&spirv, 1),
        {});
}

static Pipeline<1> makeVoxelMeshGenPipeline(const Device& dev,
    VkPipelineCache pipeline_cache,
    CountT num_frames) {
    PipelineShaders shader = makeVoxelGenShader(dev);

    // Push constant
    VkPushConstantRange push_const{
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(render::shader::VoxelGenPushConst),
    };

    // Layout configuration
    std::array desc_layouts {
        shader.getLayout(0),
    };

    VkPipelineLayoutCreateInfo voxel_gen_layout_info;
    voxel_gen_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    voxel_gen_layout_info.pNext = nullptr;
    voxel_gen_layout_info.flags = 0;
    voxel_gen_layout_info.setLayoutCount =
        static_cast<uint32_t>(desc_layouts.size());
    voxel_gen_layout_info.pSetLayouts = desc_layouts.data();
    voxel_gen_layout_info.pushConstantRangeCount = 1;
    voxel_gen_layout_info.pPushConstantRanges = &push_const;

    VkPipelineLayout voxel_gen_layout;
    REQ_VK(dev.dt.createPipelineLayout(dev.hdl, &voxel_gen_layout_info, nullptr,
        &voxel_gen_layout));

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
        "voxelGen",
        nullptr,
    };
    compute_infos[0].layout = voxel_gen_layout;
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
            voxel_gen_layout,
            pipelines,
            std::move(desc_pool),
    };
}

static PipelineShaders makeVoxelDrawShaders(
    const Device& dev, VkSampler repeat_sampler, VkSampler clamp_sampler)
{
    (void)repeat_sampler;
    (void)clamp_sampler;

    std::filesystem::path shader_dir =
        std::filesystem::path(STRINGIFY(MADRONA_RENDER_DATA_DIR)) /
        "shaders";

    auto shader_path = (shader_dir / "voxel_draw.hlsl").string();

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
    return PipelineShaders(dev, tmp_alloc, shaders,
        Span<const BindingOverride>({
            BindingOverride {
                1,
                0,
                VK_NULL_HANDLE,
                InternalConfig::maxTextures,
                VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
            },
            BindingOverride {
                1,
                1,
                repeat_sampler,
                1,
                0,
            },
            }));
}

static PipelineShaders makeShadowDrawShaders(
    const Device &dev, VkSampler repeat_sampler, VkSampler clamp_sampler)
{
    (void)repeat_sampler;
    (void)clamp_sampler;

    std::filesystem::path shader_dir =
        std::filesystem::path(STRINGIFY(MADRONA_RENDER_DATA_DIR)) /
        "shaders";

    auto shader_path = (shader_dir / "viewer_shadow_draw.hlsl").string();

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
    return PipelineShaders(dev, tmp_alloc, shaders, {});
}

static PipelineShaders makeShadowGenShader(const Device &dev, VkSampler clamp_sampler)
{
    (void)clamp_sampler;
    std::filesystem::path shader_dir =
        std::filesystem::path(STRINGIFY(MADRONA_RENDER_DATA_DIR)) /
        "shaders";

    ShaderCompiler compiler;
    SPIRVShader spirv = compiler.compileHLSLFileToSPV(
        (shader_dir / "shadow_gen.hlsl").string().c_str(), {},
        {}, { "shadowGen", ShaderStage::Compute });

    StackAlloc tmp_alloc;
    return PipelineShaders(
        dev, tmp_alloc,
        Span<const SPIRVShader>(&spirv, 1), 
        {});
}

static PipelineShaders makeDeferredLightingShader(const Device &dev, VkSampler clamp_sampler)
{
    std::filesystem::path shader_dir =
        std::filesystem::path(STRINGIFY(MADRONA_RENDER_DATA_DIR)) /
        "shaders";

    ShaderCompiler compiler;
    SPIRVShader spirv = compiler.compileHLSLFileToSPV(
        (shader_dir / "viewer_deferred_lighting.hlsl").string().c_str(), {},
        {}, { "lighting", ShaderStage::Compute });

    StackAlloc tmp_alloc;
    return PipelineShaders(
        dev, tmp_alloc,
        Span<const SPIRVShader>(&spirv, 1), 
        Span<const BindingOverride>({BindingOverride{
            0, 9, clamp_sampler, 1, 0 }}));
}

static vk::PipelineShaders makeGridDrawShader(const vk::Device &dev)
{
    std::filesystem::path shader_dir =
        std::filesystem::path(STRINGIFY(MADRONA_RENDER_DATA_DIR)) /
        "shaders";

    ShaderCompiler compiler;
    SPIRVShader spirv = compiler.compileHLSLFileToSPV(
        (shader_dir / "grid_draw.hlsl").string().c_str(), {},
        {}, {"gridDraw", ShaderStage::Compute });
    
    StackAlloc tmp_alloc;
    return vk::PipelineShaders(dev, tmp_alloc,
                               Span<const SPIRVShader>(&spirv, 1), 
                               {});
}

static PipelineShaders makeBlurShader(const Device &dev, VkSampler clamp_sampler)
{
    (void)clamp_sampler;
    std::filesystem::path shader_dir =
        std::filesystem::path(STRINGIFY(MADRONA_RENDER_DATA_DIR)) /
        "shaders";

    ShaderCompiler compiler;
    SPIRVShader spirv = compiler.compileHLSLFileToSPV(
        (shader_dir / "blur.hlsl").string().c_str(), {},
        {}, { "blur", ShaderStage::Compute });

    StackAlloc tmp_alloc;
    return PipelineShaders(
        dev, tmp_alloc,
        Span<const SPIRVShader>(&spirv, 1), 
        {});
}

static Pipeline<1> makeShadowDrawPipeline(const Device &dev,
                                    VkPipelineCache pipeline_cache,
                                    VkRenderPass render_pass,
                                    VkSampler repeat_sampler,
                                    VkSampler clamp_sampler,
                                    uint32_t num_frames)
{
    auto shaders =
        makeShadowDrawShaders(dev, repeat_sampler, clamp_sampler);

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
    depth_info.front.compareOp = VK_COMPARE_OP_ALWAYS;

    // Blend
    VkPipelineColorBlendAttachmentState blend_attach {};
    blend_attach.blendEnable = VK_FALSE;
    blend_attach.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT;

    array<VkPipelineColorBlendAttachmentState, 1> blend_attachments {{
        blend_attach,
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

    array<VkDescriptorSetLayout, 2> draw_desc_layouts {{
        shaders.getLayout(0),
        shaders.getLayout(1),
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

static Pipeline<1> makeVoxelDrawPipeline(const Device& dev,
    VkPipelineCache pipeline_cache,
    VkRenderPass render_pass,
    VkSampler repeat_sampler,
    VkSampler clamp_sampler,
    uint32_t num_frames)
{
    auto shaders =
        makeVoxelDrawShaders(dev, repeat_sampler, clamp_sampler);

    VkPipelineVertexInputStateCreateInfo vert_info{};
    VkPipelineInputAssemblyStateCreateInfo input_assembly_info{};
    VkPipelineViewportStateCreateInfo viewport_info{};
    VkPipelineMultisampleStateCreateInfo multisample_info{};
    VkPipelineRasterizationStateCreateInfo raster_info{};

    initCommonDrawPipelineInfo(vert_info, input_assembly_info,
        viewport_info, multisample_info, raster_info);

    // Depth/Stencil
    VkPipelineDepthStencilStateCreateInfo depth_info{};
    depth_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth_info.depthTestEnable = VK_TRUE;
    depth_info.depthWriteEnable = VK_TRUE;
    depth_info.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;
    depth_info.depthBoundsTestEnable = VK_FALSE;
    depth_info.stencilTestEnable = VK_FALSE;
    depth_info.back.compareOp = VK_COMPARE_OP_ALWAYS;

    // Blend
    VkPipelineColorBlendAttachmentState blend_attach{};
    blend_attach.blendEnable = VK_FALSE;
    blend_attach.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    array<VkPipelineColorBlendAttachmentState, 3> blend_attachments {{
            blend_attach,
                blend_attach,
                blend_attach
        }};

    VkPipelineColorBlendStateCreateInfo blend_info{};
    blend_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend_info.logicOpEnable = VK_FALSE;
    blend_info.attachmentCount =
        static_cast<uint32_t>(blend_attachments.size());
    blend_info.pAttachments = blend_attachments.data();

    // Dynamic
    array dyn_enable{
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };

    VkPipelineDynamicStateCreateInfo dyn_info{};
    dyn_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn_info.dynamicStateCount = dyn_enable.size();
    dyn_info.pDynamicStates = dyn_enable.data();

    // Push constant
    VkPushConstantRange push_const{
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        sizeof(DrawPushConst),
    };

    // Layout configuration

    array<VkDescriptorSetLayout, 2> draw_desc_layouts {{
            shaders.getLayout(0),
                shaders.getLayout(1)
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

static Pipeline<1> makeShadowGenPipeline(const Device &dev,
                                    VkPipelineCache pipeline_cache,
                                    VkSampler clamp_sampler,
                                    CountT num_frames)
{
    PipelineShaders shader = makeShadowGenShader(dev, clamp_sampler);

    // Push constant
    VkPushConstantRange push_const {
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(render::shader::ShadowGenPushConst),
    };

    // Layout configuration
    std::array desc_layouts {
        shader.getLayout(0),
    };

    VkPipelineLayoutCreateInfo lighting_layout_info;
    lighting_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    lighting_layout_info.pNext = nullptr;
    lighting_layout_info.flags = 0;
    lighting_layout_info.setLayoutCount =
        static_cast<uint32_t>(desc_layouts.size());
    lighting_layout_info.pSetLayouts = desc_layouts.data();
    lighting_layout_info.pushConstantRangeCount = 1;
    lighting_layout_info.pPushConstantRanges = &push_const;

    VkPipelineLayout lighting_layout;
    REQ_VK(dev.dt.createPipelineLayout(dev.hdl, &lighting_layout_info, nullptr,
                                       &lighting_layout));

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
        "shadowGen",
        nullptr,
    };
    compute_infos[0].layout = lighting_layout;
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
        lighting_layout,
        pipelines,
        std::move(desc_pool),
    };
}

static Pipeline<1> makeDeferredLightingPipeline(const Device &dev,
                                    VkPipelineCache pipeline_cache,
                                    VkSampler clamp_sampler,
                                    CountT num_frames)
{
    PipelineShaders shader = makeDeferredLightingShader(dev, clamp_sampler);

    // Push constant
    VkPushConstantRange push_const {
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(DeferredLightingPushConst),
    };

    // Layout configuration
    std::array desc_layouts {
        shader.getLayout(0),
    };

    VkPipelineLayoutCreateInfo lighting_layout_info;
    lighting_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    lighting_layout_info.pNext = nullptr;
    lighting_layout_info.flags = 0;
    lighting_layout_info.setLayoutCount =
        static_cast<uint32_t>(desc_layouts.size());
    lighting_layout_info.pSetLayouts = desc_layouts.data();
    lighting_layout_info.pushConstantRangeCount = 1;
    lighting_layout_info.pPushConstantRanges = &push_const;

    VkPipelineLayout lighting_layout;
    REQ_VK(dev.dt.createPipelineLayout(dev.hdl, &lighting_layout_info, nullptr,
                                       &lighting_layout));

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
        "lighting",
        nullptr,
    };
    compute_infos[0].layout = lighting_layout;
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
        lighting_layout,
        pipelines,
        std::move(desc_pool),
    };
}

static Pipeline<1> makeGridDrawPipeline(const Device &dev,
                                    VkPipelineCache pipeline_cache,
                                    CountT num_frames)
{
    PipelineShaders shader = makeGridDrawShader(dev);

    // Push constant
    VkPushConstantRange push_const {
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(render::shader::GridDrawPushConst),
    };

    // Layout configuration
    std::array desc_layouts {
        shader.getLayout(0),
    };

    VkPipelineLayoutCreateInfo grid_layout_info;
    grid_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    grid_layout_info.pNext = nullptr;
    grid_layout_info.flags = 0;
    grid_layout_info.setLayoutCount =
        static_cast<uint32_t>(desc_layouts.size());
    grid_layout_info.pSetLayouts = desc_layouts.data();
    grid_layout_info.pushConstantRangeCount = 1;
    grid_layout_info.pPushConstantRanges = &push_const;

    VkPipelineLayout grid_layout;
    REQ_VK(dev.dt.createPipelineLayout(dev.hdl, &grid_layout_info, nullptr,
                                       &grid_layout));

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
        "gridDraw",
        nullptr,
    };
    compute_infos[0].layout = grid_layout;
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
        grid_layout,
        pipelines,
        std::move(desc_pool),
    };
}

static PipelineShaders makeQuadShader(const Device &dev, VkSampler clamp_sampler)
{
    std::filesystem::path shader_dir =
        std::filesystem::path(STRINGIFY(MADRONA_RENDER_DATA_DIR)) /
        "shaders";

    ShaderCompiler compiler;
    SPIRVShader vert_spirv = compiler.compileHLSLFileToSPV(
        (shader_dir / "textured_quad.hlsl").string().c_str(), {},
        {}, { "vert", ShaderStage::Vertex });

    SPIRVShader frag_spirv = compiler.compileHLSLFileToSPV(
        (shader_dir / "textured_quad.hlsl").string().c_str(), {},
        {}, { "frag", ShaderStage::Fragment });

    std::array<SPIRVShader, 2> shaders {
        std::move(vert_spirv),
        std::move(frag_spirv),
    };

    StackAlloc tmp_alloc;
    return PipelineShaders(dev, tmp_alloc, shaders,
        Span<const BindingOverride>({BindingOverride{
            0, 1, clamp_sampler, 1, 0 }}));
}

static Pipeline<1> makeQuadPipeline(const Device &dev,
                                    VkPipelineCache pipeline_cache,
                                    VkSampler clamp_sampler,
                                    CountT num_frames,
                                    VkRenderPass render_pass)
{
    PipelineShaders shader = makeQuadShader(dev, clamp_sampler);

    VkPipelineVertexInputStateCreateInfo vert_info {};
    VkPipelineInputAssemblyStateCreateInfo input_assembly_info {};
    VkPipelineViewportStateCreateInfo viewport_info {};
    VkPipelineMultisampleStateCreateInfo multisample_info {};
    VkPipelineRasterizationStateCreateInfo raster_info {};

    initCommonDrawPipelineInfo(vert_info, input_assembly_info, 
        viewport_info, multisample_info, raster_info);

    raster_info.cullMode = VK_CULL_MODE_NONE;
    input_assembly_info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;

    // Depth/Stencil
    VkPipelineDepthStencilStateCreateInfo depth_info {};
    depth_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth_info.depthTestEnable = VK_FALSE;
    depth_info.depthWriteEnable = VK_FALSE;
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

    array<VkPipelineColorBlendAttachmentState, 1> blend_attachments {{
        blend_attach,
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
        sizeof(render::shader::TexturedQuadPushConst),
    };

    // Layout configuration

    array<VkDescriptorSetLayout, 1> draw_desc_layouts {{
        shader.getLayout(0),
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
            shader.getShader(0),
            "vert",
            nullptr,
        },
        {
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            nullptr,
            0,
            VK_SHADER_STAGE_FRAGMENT_BIT,
            shader.getShader(1),
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

    FixedDescriptorPool desc_pool(dev, shader, 0, num_frames);

    return {
        std::move(shader),
        draw_layout,
        { draw_pipeline },
        std::move(desc_pool),
    };
}

static Pipeline<1> makeBlurPipeline(const Device &dev,
                                    VkPipelineCache pipeline_cache,
                                    VkSampler clamp_sampler,
                                    CountT num_frames)
{
    PipelineShaders shader = makeBlurShader(dev, clamp_sampler);

    // Push constant
    VkPushConstantRange push_const {
        VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        sizeof(render::shader::BlurPushConst),
    };

    // Layout configuration
    std::array desc_layouts {
        shader.getLayout(0),
    };

    VkPipelineLayoutCreateInfo blur_layout_info;
    blur_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    blur_layout_info.pNext = nullptr;
    blur_layout_info.flags = 0;
    blur_layout_info.setLayoutCount =
        static_cast<uint32_t>(desc_layouts.size());
    blur_layout_info.pSetLayouts = desc_layouts.data();
    blur_layout_info.pushConstantRangeCount = 1;
    blur_layout_info.pPushConstantRanges = &push_const;

    VkPipelineLayout blur_layout;
    REQ_VK(dev.dt.createPipelineLayout(dev.hdl, &blur_layout_info, nullptr,
                                       &blur_layout));

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
        "blur",
        nullptr,
    };
    compute_infos[0].layout = blur_layout;
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
        blur_layout,
        pipelines,
        std::move(desc_pool),
    };
}


static array<VkClearValue, 2> makeImguiClearValues()
{
    VkClearValue color_clear;
    color_clear.color = {{0.f, 0.f, 0.f, 1.f}};

    VkClearValue depth_clear;
    depth_clear.depthStencil = {0.f, 0};

    return {
        color_clear,
        depth_clear,
    };
}

static array<VkClearValue, 4> makeClearValues()
{
    VkClearValue color_clear;
    color_clear.color = {{0.f, 0.f, 0.f, 0.f}};

    VkClearValue depth_clear;
    depth_clear.depthStencil = {0.f, 0};

    return {
        color_clear,
        color_clear,
        color_clear,
        depth_clear,
    };
}

static array<VkClearValue, 2> makeShadowClearValues()
{
    VkClearValue color_clear = {};
    color_clear.color = {{0.f, 0.f, 0.f, 0.f}};

    VkClearValue depth_clear = {};
    depth_clear.depthStencil = {0.f, 0};

    return {
        depth_clear,
    };
}

static void destroyFramebuffers(const Device &dev,
                                MemoryAllocator &alloc,
                                Framebuffer &fb,
                                Framebuffer &imgui_fb)
{
    (void)alloc;

    dev.dt.destroyFramebuffer(dev.hdl, fb.hdl, nullptr);
    dev.dt.destroyFramebuffer(dev.hdl, imgui_fb.hdl, nullptr);
    dev.dt.destroyImageView(dev.hdl, fb.colorView, nullptr);
    dev.dt.destroyImageView(dev.hdl, fb.normalView, nullptr);
    dev.dt.destroyImageView(dev.hdl, fb.positionView, nullptr);
    dev.dt.destroyImageView(dev.hdl, fb.depthView, nullptr);

    // This sucks - will fix momentarily once the imgui refactor comes in
    fb.colorAttachment.~LocalImage();
    fb.normalAttachment.~LocalImage();
    fb.positionAttachment.~LocalImage();
    fb.depthAttachment.~LocalImage();
}

static std::pair<Framebuffer, Framebuffer> makeFramebuffers(
    const Device &dev,
    MemoryAllocator &alloc,
    uint32_t fb_width,
    uint32_t fb_height,
    VkRenderPass render_pass,
    VkRenderPass imgui_render_pass)
{
    auto albedo = alloc.makeColorAttachment(
        fb_width, fb_height, 1, VK_FORMAT_R8G8B8A8_UNORM);
    auto normal = alloc.makeColorAttachment(
        fb_width, fb_height, 1, VK_FORMAT_R16G16B16A16_SFLOAT);
    auto position = alloc.makeColorAttachment(
        fb_width, fb_height, 1, VK_FORMAT_R16G16B16A16_SFLOAT);
    auto depth = alloc.makeDepthAttachment(
        fb_width, fb_height, 1, InternalConfig::depthFormat);

    VkImageViewCreateInfo view_info {};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    VkImageSubresourceRange &view_info_sr = view_info.subresourceRange;
    view_info_sr.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info_sr.baseMipLevel = 0;
    view_info_sr.levelCount = 1;
    view_info_sr.baseArrayLayer = 0;
    view_info_sr.layerCount = 1;

    view_info.image = albedo.image;
    view_info.format = VK_FORMAT_R8G8B8A8_UNORM;

    VkImageView albedo_view;
    REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &albedo_view));

    view_info.image = normal.image;
    view_info.format = VK_FORMAT_R16G16B16A16_SFLOAT;

    VkImageView normal_view;
    REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &normal_view));

    view_info.image = position.image;
    view_info.format = VK_FORMAT_R16G16B16A16_SFLOAT;

    VkImageView position_view;
    REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &position_view));

    view_info.image = depth.image;
    view_info.format = InternalConfig::depthFormat;
    view_info_sr.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

    VkImageView depth_view;
    REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &depth_view));

    array attachment_views {
        albedo_view,
        normal_view,
        position_view,
        depth_view,
    };

    VkFramebufferCreateInfo fb_info;
    fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fb_info.pNext = nullptr;
    fb_info.flags = 0;
    fb_info.renderPass = render_pass;
    fb_info.attachmentCount = static_cast<uint32_t>(attachment_views.size());
    fb_info.pAttachments = attachment_views.data();
    fb_info.width = fb_width;
    fb_info.height = fb_height;
    fb_info.layers = 1;

    VkFramebuffer hdl;
    REQ_VK(dev.dt.createFramebuffer(dev.hdl, &fb_info, nullptr, &hdl));

    array imgui_attachment_views {
        albedo_view,
        depth_view,
    };

    VkFramebufferCreateInfo imgui_fb_info;
    imgui_fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    imgui_fb_info.pNext = nullptr;
    imgui_fb_info.flags = 0;
    imgui_fb_info.renderPass = imgui_render_pass;
    imgui_fb_info.attachmentCount = 2;
    imgui_fb_info.pAttachments = imgui_attachment_views.data();
    imgui_fb_info.width = fb_width;
    imgui_fb_info.height = fb_height;
    imgui_fb_info.layers = 1;

    VkFramebuffer imgui_hdl;
    REQ_VK(dev.dt.createFramebuffer(dev.hdl, &imgui_fb_info, nullptr, &imgui_hdl));

    return std::make_pair(
        Framebuffer {
            std::move(albedo),
            std::move(normal),
            std::move(position),
            std::move(depth),
            albedo_view,
            normal_view,
            position_view,
            depth_view,
            hdl 
        },
        Framebuffer {
            LocalImage::makeEmpty(),
            LocalImage::makeEmpty(),
            LocalImage::makeEmpty(),
            LocalImage::makeEmpty(),
            VK_NULL_HANDLE,
            VK_NULL_HANDLE,
            VK_NULL_HANDLE,
            VK_NULL_HANDLE,
            imgui_hdl 
        });
}

static ShadowFramebuffer makeShadowFramebuffer(const Device &dev,
                                   MemoryAllocator &alloc,
                                   uint32_t fb_width,
                                   uint32_t fb_height,
                                   VkRenderPass render_pass)
{
    auto color = alloc.makeColorAttachment(fb_width, fb_height, 1,
        InternalConfig::varianceFormat);
    auto intermediate = alloc.makeColorAttachment(fb_width, fb_height, 1,
        InternalConfig::varianceFormat);
    auto depth = alloc.makeDepthAttachment(
        fb_width, fb_height, 1, InternalConfig::depthFormat);

    VkImageViewCreateInfo view_info {};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    VkImageSubresourceRange &view_info_sr = view_info.subresourceRange;
    view_info_sr.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    view_info_sr.baseMipLevel = 0;
    view_info_sr.levelCount = 1;
    view_info_sr.baseArrayLayer = 0;
    view_info_sr.layerCount = 1;

    view_info.image = depth.image;
    view_info.format = InternalConfig::depthFormat;

    VkImageView depth_view;
    REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &depth_view));

    view_info_sr.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.image = color.image;
    view_info.format = InternalConfig::varianceFormat;

    VkImageView color_view;
    REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &color_view));

    view_info.image = intermediate.image;
    view_info.format = InternalConfig::varianceFormat;

    VkImageView intermediate_view;
    REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &intermediate_view));

    array attachment_views {
        color_view,
        depth_view
    };

    VkFramebufferCreateInfo fb_info;
    fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fb_info.pNext = nullptr;
    fb_info.flags = 0;
    fb_info.renderPass = render_pass;
    fb_info.attachmentCount = static_cast<uint32_t>(attachment_views.size());
    fb_info.pAttachments = attachment_views.data();
    fb_info.width = fb_width;
    fb_info.height = fb_height;
    fb_info.layers = 1;

    VkFramebuffer hdl;
    REQ_VK(dev.dt.createFramebuffer(dev.hdl, &fb_info, nullptr, &hdl));

    return ShadowFramebuffer {
        std::move(color),
        std::move(intermediate),
        std::move(depth),
        color_view,
        intermediate_view,
        depth_view,
        hdl,
    };
}

static void makeFrame(Frame *dst,
                      const Device &dev, MemoryAllocator &alloc,
                      uint32_t fb_width, uint32_t fb_height,
                      uint32_t max_views, uint32_t max_instances,
                      VoxelConfig voxel_config,
                      VkRenderPass render_pass,
                      VkRenderPass imgui_render_pass,
                      VkRenderPass shadow_pass,
                      VkDescriptorSet cull_set,
                      VkDescriptorSet draw_set,
                      VkDescriptorSet lighting_set,
                      VkDescriptorSet shadow_gen_set,
                      VkDescriptorSet shadow_blur_set,
                      VkDescriptorSet voxel_gen_set,
                      VkDescriptorSet voxel_draw_set,
                      VkDescriptorSet quad_draw_set,
                      VkDescriptorSet grid_draw_set,
                      Sky &sky,
                      BatchImportedBuffers &batch_renderer_buffers,
                      const LocalBuffer &batch_rgb_out,
                      const LocalBuffer &batch_depth_out)
{
    auto [fb, imgui_fb] = makeFramebuffers(dev, alloc, 
            fb_width, fb_height, render_pass, imgui_render_pass);

    auto shadow_fb = makeShadowFramebuffer(dev, alloc, 
        InternalConfig::shadowMapSize,
        InternalConfig::shadowMapSize,
        shadow_pass);

    VkCommandPool render_viewer_cmd_pool = makeCmdPool(dev, dev.gfxQF);
    VkCommandPool present_viewer_cmd_pool = makeCmdPool(dev, dev.gfxQF);

    int64_t buffer_offsets[6];
    int64_t buffer_sizes[7] = {
        // We just store the flycam view here
        (int64_t)sizeof(PackedViewData),
        (int64_t)sizeof(uint32_t),
        // (int64_t)sizeof(PackedInstanceData) * max_instances,
        (int64_t)sizeof(DrawCmd) * max_instances * 10,
        (int64_t)sizeof(DrawData) * max_instances * 10,
        (int64_t)sizeof(DirectionalLight) * InternalConfig::maxLights,
        (int64_t)sizeof(ShadowViewData) * (max_views + 1),
        (int64_t)sizeof(SkyData)
    };

    int64_t num_render_input_bytes = utils::computeBufferOffsets(
        buffer_sizes, buffer_offsets, 256);

    HostBuffer view_staging = alloc.makeStagingBuffer(sizeof(PackedViewData));
    HostBuffer light_staging = alloc.makeStagingBuffer(sizeof(DirectionalLight) * InternalConfig::maxLights);
    // HostBuffer shadow_staging = alloc.makeStagingBuffer(sizeof(ShadowViewData));
    HostBuffer sky_staging = alloc.makeStagingBuffer(sizeof(SkyData));

    LocalBuffer render_input = *alloc.makeLocalBuffer(num_render_input_bytes);

    std::array<VkWriteDescriptorSet, 34> desc_updates;
    uint32_t desc_counter = 0;

    VkDescriptorBufferInfo view_info;
    view_info.buffer = render_input.buffer;
    view_info.offset = 0;
    view_info.range = buffer_sizes[0];

    // Right now, the view_info isn't used in the cull shader
    // DescHelper::uniform(desc_updates[23], cull_set, &view_info, 0);
    DescHelper::storage(desc_updates[desc_counter++], draw_set, &view_info, 0);
    DescHelper::storage(desc_updates[desc_counter++], shadow_gen_set, &view_info, 1);

    VkDescriptorBufferInfo instance_info;
    instance_info.buffer = batch_renderer_buffers.instances.buffer;
    instance_info.offset = 0;
    instance_info.range = VK_WHOLE_SIZE;

    DescHelper::storage(desc_updates[desc_counter++], cull_set, &instance_info, 1);
    DescHelper::storage(desc_updates[desc_counter++], draw_set, &instance_info, 1);

    VkDescriptorBufferInfo instance_offset_info;
    instance_offset_info.buffer = batch_renderer_buffers.instanceOffsets.buffer;
    instance_offset_info.offset = 0;
    instance_offset_info.range = VK_WHOLE_SIZE;



    VkDescriptorBufferInfo batch_view_info;
    batch_view_info.buffer = batch_renderer_buffers.views.buffer;
    batch_view_info.offset = 0;
    batch_view_info.range = VK_WHOLE_SIZE;

    DescHelper::storage(desc_updates[desc_counter++], draw_set, &batch_view_info, 4);
    DescHelper::storage(desc_updates[desc_counter++], shadow_gen_set, &batch_view_info, 3);

    VkDescriptorBufferInfo batch_view_offset_info;
    batch_view_offset_info.buffer = batch_renderer_buffers.viewOffsets.buffer;
    batch_view_offset_info.offset = 0;
    batch_view_offset_info.range = VK_WHOLE_SIZE;

    DescHelper::storage(desc_updates[desc_counter++], draw_set, &batch_view_offset_info, 5);
    DescHelper::storage(desc_updates[desc_counter++], shadow_gen_set, &batch_view_offset_info, 4);



    DescHelper::storage(desc_updates[desc_counter++], cull_set, &instance_offset_info, 5);

    VkDescriptorBufferInfo drawcount_info;
    drawcount_info.buffer = render_input.buffer;
    drawcount_info.offset = buffer_offsets[0];
    drawcount_info.range = buffer_sizes[1];

    DescHelper::storage(desc_updates[desc_counter++], cull_set, &drawcount_info, 2);

    VkDescriptorBufferInfo draw_info;
    draw_info.buffer = render_input.buffer;
    draw_info.offset = buffer_offsets[1];
    draw_info.range = buffer_sizes[2];

    DescHelper::storage(desc_updates[desc_counter++], cull_set, &draw_info, 3);

    VkDescriptorBufferInfo draw_data_info;
    draw_data_info.buffer = render_input.buffer;
    draw_data_info.offset = buffer_offsets[2];
    draw_data_info.range = buffer_sizes[3];

    DescHelper::storage(desc_updates[desc_counter++], cull_set, &draw_data_info, 4);
    DescHelper::storage(desc_updates[desc_counter++], draw_set, &draw_data_info, 2);

    VkDescriptorImageInfo gbuffer_albedo_info;
    gbuffer_albedo_info.imageView = fb.colorView;
    gbuffer_albedo_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    gbuffer_albedo_info.sampler = VK_NULL_HANDLE;

    DescHelper::storageImage(desc_updates[desc_counter++], lighting_set, &gbuffer_albedo_info, 0);
    
    if (grid_draw_set != VK_NULL_HANDLE) {
        DescHelper::storageImage(desc_updates[desc_counter++], grid_draw_set, &gbuffer_albedo_info, 0);
    }

    VkDescriptorImageInfo gbuffer_normal_info;
    gbuffer_normal_info.imageView = fb.normalView;
    gbuffer_normal_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    gbuffer_normal_info.sampler = VK_NULL_HANDLE;

    DescHelper::storageImage(desc_updates[desc_counter++], lighting_set, &gbuffer_normal_info, 1);

    VkDescriptorImageInfo gbuffer_position_info;
    gbuffer_position_info.imageView = fb.positionView;
    gbuffer_position_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    gbuffer_position_info.sampler = VK_NULL_HANDLE;

    DescHelper::storageImage(desc_updates[desc_counter++], lighting_set, &gbuffer_position_info, 2);

    VkDescriptorBufferInfo light_data_info;
    light_data_info.buffer = render_input.buffer;
    light_data_info.offset = buffer_offsets[3];
    light_data_info.range = buffer_sizes[4];

    DescHelper::storage(desc_updates[desc_counter++], lighting_set, &light_data_info, 3);
    DescHelper::storage(desc_updates[desc_counter++], shadow_gen_set, &light_data_info, 2);

    VkDescriptorImageInfo transmittance_info;
    transmittance_info.imageView = sky.transmittanceView;
    transmittance_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    transmittance_info.sampler = VK_NULL_HANDLE;

    DescHelper::textures(desc_updates[desc_counter++], lighting_set, &transmittance_info, 1, 4, 0);

    VkDescriptorImageInfo irradiance_info;
    irradiance_info.imageView = sky.irradianceView;
    irradiance_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    irradiance_info.sampler = VK_NULL_HANDLE;

    DescHelper::textures(desc_updates[desc_counter++], lighting_set, &irradiance_info, 1, 5, 0);

    VkDescriptorImageInfo scattering_info;
    scattering_info.imageView = sky.scatteringView;
    scattering_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    scattering_info.sampler = VK_NULL_HANDLE;

    DescHelper::textures(desc_updates[desc_counter++], lighting_set, &scattering_info, 1, 6, 0);

    VkDescriptorBufferInfo shadow_view_info;
    shadow_view_info.buffer = render_input.buffer;
    shadow_view_info.offset = buffer_offsets[4];
    shadow_view_info.range = buffer_sizes[5];

    DescHelper::storage(desc_updates[desc_counter++], draw_set, &shadow_view_info, 3);
    DescHelper::storage(desc_updates[desc_counter++], lighting_set, &shadow_view_info, 8);
    DescHelper::storage(desc_updates[desc_counter++], shadow_gen_set, &shadow_view_info, 0);

    VkDescriptorImageInfo shadow_map_info;
    shadow_map_info.imageView = shadow_fb.varianceView;
    // shadow_map_info.imageView = shadow_fb.depthView;
    shadow_map_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    shadow_map_info.sampler = VK_NULL_HANDLE;

    DescHelper::textures(desc_updates[desc_counter++], lighting_set, &shadow_map_info, 1, 7, 0);

    VkDescriptorBufferInfo sky_info;
    sky_info.buffer = render_input.buffer;
    sky_info.offset = buffer_offsets[5];
    sky_info.range = buffer_sizes[6];

    DescHelper::storage(desc_updates[desc_counter++], lighting_set, &sky_info, 10);

    VkDescriptorImageInfo blur_input_info;
    blur_input_info.imageView = shadow_fb.varianceView;
    blur_input_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    blur_input_info.sampler = VK_NULL_HANDLE;

    DescHelper::storageImage(desc_updates[desc_counter++], shadow_blur_set, &blur_input_info, 0);

    VkDescriptorImageInfo blur_intermediate_info;
    blur_intermediate_info.imageView = shadow_fb.intermediateView;
    blur_intermediate_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    blur_intermediate_info.sampler = VK_NULL_HANDLE;

    DescHelper::storageImage(desc_updates[desc_counter++], shadow_blur_set, &blur_intermediate_info, 1);


    //Voxelizer Changes
    const int32_t num_voxels = voxel_config.xLength * voxel_config.yLength * voxel_config.zLength;
    const int32_t voxels_size = num_voxels > 0 ? sizeof(int32_t) * num_voxels : 4;
    const int32_t vertices_size = num_voxels > 0 ? num_voxels * 32 * 6 * sizeof(float) : 4;
    const int32_t indices_size = num_voxels > 0 ? num_voxels * 6 * 6 * sizeof(int32_t) : 4;

    std::array<VkWriteDescriptorSet, 7> voxel_updates;

    LocalBuffer voxel_vbo = *alloc.makeLocalBuffer(vertices_size);
    LocalBuffer voxel_ibo = *alloc.makeLocalBuffer(indices_size);
    LocalBuffer voxel_data = *alloc.makeLocalBuffer(voxels_size);

    VkDescriptorBufferInfo voxel_vbo_info;
    voxel_vbo_info.buffer = voxel_vbo.buffer;
    voxel_vbo_info.offset = 0;
    voxel_vbo_info.range = vertices_size;

    DescHelper::storage(voxel_updates[0], voxel_gen_set, &voxel_vbo_info, 0);

    VkDescriptorBufferInfo voxelIndexBuffer_info;
    voxelIndexBuffer_info.buffer = voxel_ibo.buffer;
    voxelIndexBuffer_info.offset = 0;
    voxelIndexBuffer_info.range = indices_size;

    DescHelper::storage(voxel_updates[1], voxel_gen_set, &voxelIndexBuffer_info, 1);

    VkDescriptorBufferInfo voxel_info;
    voxel_info.buffer = voxel_data.buffer;
    voxel_info.offset = 0;
    voxel_info.range = voxels_size;

    DescHelper::storage(voxel_updates[2], voxel_gen_set, &voxel_info, 2);

    DescHelper::storage(voxel_updates[3], voxel_draw_set, &view_info, 0);

    VkDescriptorBufferInfo vert_info;
    vert_info.buffer = voxel_vbo.buffer;
    vert_info.offset = 0;
    vert_info.range = vertices_size;
    DescHelper::storage(voxel_updates[4], voxel_draw_set, &vert_info, 1);

    DescHelper::storage(voxel_updates[5], voxel_draw_set, &batch_view_info, 2);
    DescHelper::storage(voxel_updates[6], voxel_draw_set, &batch_view_offset_info, 3);

    DescHelper::update(dev, voxel_updates.data(), voxel_updates.size());

    DescHelper::update(dev, desc_updates.data(), desc_counter);

    if (grid_draw_set != VK_NULL_HANDLE) {
        // Create the descriptor set with the batch renderer outputs
        VkDescriptorBufferInfo batch_rgb_info {
            .buffer = batch_rgb_out.buffer,
            .offset = 0,
            .range = VK_WHOLE_SIZE,
        };

        VkDescriptorBufferInfo batch_depth_info {
            .buffer = batch_depth_out.buffer,
            .offset = 0,
            .range = VK_WHOLE_SIZE,
        };

        std::array<VkWriteDescriptorSet, 2> batch_out_updates;
        vk::DescHelper::storage(batch_out_updates[0], grid_draw_set,
                                &batch_rgb_info, 1);
        vk::DescHelper::storage(batch_out_updates[1], grid_draw_set,
                                &batch_depth_info, 2);
        vk::DescHelper::update(
            dev, batch_out_updates.data(), batch_out_updates.size());
    }

    new (dst) Frame {
        std::move(fb),
        std::move(imgui_fb),
        std::move(shadow_fb),
        render_viewer_cmd_pool,
        makeCmdBuffer(dev, render_viewer_cmd_pool),
        present_viewer_cmd_pool,
        makeCmdBuffer(dev, present_viewer_cmd_pool),

        makeFence(dev, true),
        makeBinarySemaphore(dev),
        makeBinarySemaphore(dev),
        makeBinarySemaphore(dev),

        std::move(view_staging),
        std::move(light_staging),
        std::move(sky_staging),
        std::move(render_input),
        std::move(voxel_vbo),
        std::move(voxel_ibo),
        std::move(voxel_data),
        num_render_input_bytes,

        0,
        // sizeof(PackedViewData),
        uint32_t(buffer_offsets[1]),
        uint32_t(buffer_offsets[0]),
        // uint32_t(buffer_offsets[1]),
        (uint32_t)buffer_offsets[3],
        (uint32_t)buffer_offsets[4],
        (uint32_t)buffer_offsets[5],
        max_instances * 10,

        cull_set,
        draw_set,
        lighting_set,
        shadow_gen_set,
        shadow_blur_set,
        voxel_gen_set,
        voxel_draw_set,
        quad_draw_set,
        grid_draw_set
    };
}


static void packView(const Device &dev,
                     HostBuffer &view_staging_buffer,
                     const viz::ViewerCam &cam,
                     uint32_t fb_width, uint32_t fb_height)
{
    PackedViewData *staging = (PackedViewData *)view_staging_buffer.ptr;

    math::Quat rotation =
        math::Quat::fromBasis(cam.right, cam.fwd, cam.up).inv();

    float fov_scale = 1.f / tanf(math::toRadians(cam.fov * 0.5f));
    float aspect = float(fb_width) / float(fb_height);

    float x_scale = fov_scale / aspect;
    float y_scale = -fov_scale;

    math::Vector4 d0 {
        cam.position.x,
        cam.position.y,
        cam.position.z,
        rotation.w,
    };

    math::Vector4 d1 {
        rotation.x,
        rotation.y,
        rotation.z,
        x_scale,
    };

    math::Vector4 d2 {
        y_scale,
        0.001f,
        0.f,
        0.f,
    };

    staging->data[0] = d0;
    staging->data[1] = d1;
    staging->data[2] = d2;

    view_staging_buffer.flush(dev);
}

static void packSky( const Device &dev,
                     HostBuffer &staging)
{
    SkyData *data = (SkyData *)staging.ptr;

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
        DensityLayer { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, {} };
    data->rayleighDensity.layers[1] =
        DensityLayer { 0.0f, 1.0f, -0.125f, 0.0f, 0.0f, {} };
    data->rayleighScatteringCoef =
        math::Vector4{0.005802f, 0.013558f, 0.033100f, 0.0f};

    data->mieDensity.layers[0] =
        DensityLayer { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, {} };
    data->mieDensity.layers[1] =
        DensityLayer { 0.0f, 1.0f, -0.833333f, 0.0f, 0.0f, {} };
    data->mieScatteringCoef = math::Vector4{0.003996f, 0.003996f, 0.003996f, 0.0f};
    data->mieExtinctionCoef = math::Vector4{0.004440f, 0.004440f, 0.004440f, 0.0f};

    data->miePhaseFunctionG = 0.8f;

    data->absorptionDensity.layers[0] =
        DensityLayer { 25.000000f, 0.000000f, 0.000000f, 0.066667f, -0.666667f, {} };
    data->absorptionDensity.layers[1] =
        DensityLayer { 0.000000f, 0.000000f, 0.000000f, -0.066667f, 2.666667f, {} };
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

static void packLighting(const Device &dev,
                         HostBuffer &light_staging_buffer,
                         const HeapArray<DirectionalLight> &lights)
{
    DirectionalLight *staging = (DirectionalLight *)light_staging_buffer.ptr;
    memcpy(staging, lights.data(),
           sizeof(DirectionalLight) * InternalConfig::maxLights);
    light_staging_buffer.flush(dev);
}

static void issueGridDrawPass(RenderContext &rctx,
                              Frame &frame, 
                              Pipeline<1> &pipeline,
                              VkCommandBuffer draw_cmd,
                              const viz::ViewerControl &viz_ctrl)
{
    auto &dev = rctx.dev;
    uint32_t num_views = *rctx.engine_interop_.bridge.totalNumViews;

    {
        array<VkImageMemoryBarrier, 1> compute_prepare {{
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_NONE,
                VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                frame.fb.colorAttachment.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            }
        }};

        dev.dt.cmdPipelineBarrier(draw_cmd,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0, nullptr, 0, nullptr,
                compute_prepare.size(), compute_prepare.data());
    }

    dev.dt.cmdBindPipeline(
        draw_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.hdls[0]);

    render::shader::GridDrawPushConst push_const = {
        .numViews = num_views,
        .viewWidth = rctx.br_width_,
        .viewHeight = rctx.br_height_,
        .gridWidth = viz_ctrl.gridWidth,
        .gridViewSize = viz_ctrl.gridImageSize,
        .maxViewsPerImage = dev.maxNumLayersPerImage,
        .offsetX = viz_ctrl.batchRenderOffsetX,
        .offsetY = viz_ctrl.batchRenderOffsetY,
        .showDepth = viz_ctrl.batchRenderShowDepth ? 1_u32 : 0_u32,
    };

    dev.dt.cmdPushConstants(draw_cmd, pipeline.layout,
        VK_SHADER_STAGE_COMPUTE_BIT, 0,
        sizeof(render::shader::GridDrawPushConst), &push_const);

    dev.dt.cmdBindDescriptorSets(draw_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.layout, 0, 1, &frame.gridDrawSet, 0, nullptr);

    uint32_t num_workgroups_x = utils::divideRoundUp(
        frame.fb.colorAttachment.width, 32_u32);
    uint32_t num_workgroups_y = utils::divideRoundUp(
        frame.fb.colorAttachment.height, 32_u32);

    dev.dt.cmdDispatch(draw_cmd, num_workgroups_x, num_workgroups_y, 1);

    {
        array<VkImageMemoryBarrier, 1> compute_prepare {{
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                frame.fb.colorAttachment.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            }
        }};

        dev.dt.cmdPipelineBarrier(draw_cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                0,
                0, nullptr, 0, nullptr,
                compute_prepare.size(), compute_prepare.data());
    }
}

static void issueCulling(Device &dev,
                         VkCommandBuffer draw_cmd,
                         const Frame &frame,
                         const Pipeline<1> &instance_cull,
                         VkDescriptorSet asset_set_cull,
                         uint32_t world_idx,
                         uint32_t num_instances,
                         uint32_t num_views,
                         uint32_t num_worlds)
{
    dev.dt.cmdBindPipeline(draw_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           instance_cull.hdls[0]);

    std::array cull_descriptors {
        frame.cullShaderSet,
        asset_set_cull,
    };

    dev.dt.cmdBindDescriptorSets(draw_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 instance_cull.layout, 0,
                                 cull_descriptors.size(),
                                 cull_descriptors.data(),
                                 0, nullptr);

    uint32_t num_warps = 4;

    CullPushConst cull_push_const {
        world_idx,
        num_views,
        num_instances,
        num_worlds,
        num_warps * 32
    };

    dev.dt.cmdPushConstants(draw_cmd, instance_cull.layout,
                            VK_SHADER_STAGE_COMPUTE_BIT, 0,
                            sizeof(CullPushConst), &cull_push_const);

    // Just spawn 4 for now - we don't know how many instances to process
    dev.dt.cmdDispatch(draw_cmd, num_warps, 1, 1);

    VkMemoryBarrier cull_draw_barrier {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = 
            VK_ACCESS_INDIRECT_COMMAND_READ_BIT |
            VK_ACCESS_SHADER_READ_BIT,
    };

    dev.dt.cmdPipelineBarrier(draw_cmd,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT |
                                VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                              0, 1, &cull_draw_barrier, 0, nullptr,
                              0, nullptr);
}

static void issueLightingPass(vk::Device &dev,
                              Frame &frame, 
                              Pipeline<1> &pipeline,
                              VkCommandBuffer draw_cmd,
                              const viz::ViewerCam &cam,
                              uint32_t view_idx,
                              uint32_t world_idx)
{
    { // Transition for compute
        array<VkImageMemoryBarrier, 3> compute_prepare {{
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                frame.fb.colorAttachment.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            },
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                frame.fb.normalAttachment.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            },
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                frame.fb.positionAttachment.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
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

    DeferredLightingPushConst push_const = {
        math::Vector4{ cam.fwd.x, cam.fwd.y, cam.fwd.z, 0.0f },
        math::Vector4{ cam.position.x, cam.position.y, cam.position.z, 0.0f },
        math::toRadians(cam.fov), 20.0f, 50.0f, view_idx, world_idx
    };

    dev.dt.cmdPushConstants(draw_cmd, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DeferredLightingPushConst), &push_const);

    dev.dt.cmdBindDescriptorSets(draw_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.layout, 0, 1, &frame.lightingSet, 0, nullptr);

    uint32_t num_workgroups_x = utils::divideRoundUp(frame.fb.colorAttachment.width, 32_u32);
    uint32_t num_workgroups_y = utils::divideRoundUp(frame.fb.colorAttachment.height, 32_u32);

    dev.dt.cmdDispatch(draw_cmd, num_workgroups_x, num_workgroups_y, 1);

    { // Transition for compute
        array<VkImageMemoryBarrier, 3> compute_prepare {{
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                frame.fb.colorAttachment.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            },
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                frame.fb.normalAttachment.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            },
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                frame.fb.positionAttachment.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
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

static void issueShadowBlurPass(vk::Device &dev, Frame &frame, Pipeline<1> &pipeline, VkCommandBuffer draw_cmd)
{
    { // Transition for compute
        array<VkImageMemoryBarrier, 2> compute_prepare {{
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                frame.shadowFB.varianceAttachment.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            },
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_NONE,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                frame.shadowFB.intermediate.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
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

    render::shader::BlurPushConst push_const;
    push_const.isVertical = 0;

    dev.dt.cmdPushConstants(draw_cmd, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_const), &push_const);

    dev.dt.cmdBindDescriptorSets(draw_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.layout, 0, 1, &frame.shadowBlurSet, 0, nullptr);

    uint32_t num_workgroups_x = utils::divideRoundUp(InternalConfig::shadowMapSize, 32_u32);
    uint32_t num_workgroups_y = utils::divideRoundUp(InternalConfig::shadowMapSize, 32_u32);

    dev.dt.cmdDispatch(draw_cmd, num_workgroups_x, num_workgroups_y, 1);

    { // Transition for compute
        array<VkImageMemoryBarrier, 2> compute_prepare {{
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                frame.shadowFB.varianceAttachment.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            },
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                frame.shadowFB.intermediate.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            },
        }};

        dev.dt.cmdPipelineBarrier(draw_cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0, nullptr, 0, nullptr,
                compute_prepare.size(), compute_prepare.data());
    }

    dev.dt.cmdBindPipeline(draw_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.hdls[0]);

    push_const.isVertical = 1;

    dev.dt.cmdPushConstants(draw_cmd, pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_const), &push_const);

    dev.dt.cmdBindDescriptorSets(draw_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.layout, 0, 1, &frame.shadowBlurSet, 0, nullptr);

    dev.dt.cmdDispatch(draw_cmd, num_workgroups_x, num_workgroups_y, 1);

    { // Transition for compute
        array<VkImageMemoryBarrier, 2> compute_prepare {{
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                frame.shadowFB.varianceAttachment.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            },
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                frame.shadowFB.intermediate.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            },
        }};

        dev.dt.cmdPipelineBarrier(draw_cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0, nullptr, 0, nullptr,
                compute_prepare.size(), compute_prepare.data());
    }
}

static void issueVoxelGen(Device& dev,
    Frame& frame,
    Pipeline<1>& pipeline,
    VkCommandBuffer draw_cmd,
    uint32_t view_idx,
    uint32_t max_views,
    VoxelConfig voxel_config)
{
    (void)view_idx, (void)max_views;

    const uint32_t num_voxels = voxel_config.xLength * voxel_config.yLength * voxel_config.zLength;

    {
        VkBufferMemoryBarrier compute_prepare = {
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            nullptr,
           VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT,
            VK_ACCESS_MEMORY_WRITE_BIT|VK_ACCESS_MEMORY_READ_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            frame.voxelData.buffer,
            0,
            (VkDeviceSize)(num_voxels * sizeof(uint32_t))
        };

        dev.dt.cmdPipelineBarrier(draw_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 1, &compute_prepare, 0, nullptr);
    }

    {
        VkBufferMemoryBarrier compute_prepare = {
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            nullptr,
           VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT,
            VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            frame.voxelVBO.buffer,
            0,
            (VkDeviceSize)(32 * 4 * 6 * num_voxels)
        };

        dev.dt.cmdPipelineBarrier(draw_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 1, &compute_prepare, 0, nullptr);
    }


    dev.dt.cmdBindPipeline(draw_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.hdls[0]);

    render::shader::VoxelGenPushConst push_const = {voxel_config.xLength,voxel_config.yLength,voxel_config.zLength, 0.8, 9 };

    dev.dt.cmdPushConstants(draw_cmd,
        pipeline.layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(render::shader::VoxelGenPushConst),
        &push_const);

    dev.dt.cmdBindDescriptorSets(draw_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline.layout, 0, 1, &frame.voxelGenSet, 0, nullptr);

    //uint32_t num_workgroups_x = utils::divideRoundUp(max_views, 32_u32);
    dev.dt.cmdDispatch(draw_cmd, 64, 1, 1);

    {
        VkBufferMemoryBarrier compute_prepare = {
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT,
            VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            frame.voxelVBO.buffer,
            0,
            (VkDeviceSize)(32 * 4 * 6 * num_voxels)
        };

        dev.dt.cmdPipelineBarrier(draw_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
            0, 0, nullptr, 1, &compute_prepare, 0, nullptr);
    }
    /* {
        VkBufferMemoryBarrier compute_prepare = {
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT,
            VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            frame.voxelIndexBuffer.buffer,
            0,
            (VkDeviceSize)(6 *6* 4 * 50 * 50 * 50)
        };

        dev.dt.cmdPipelineBarrier(draw_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
            0, 0, nullptr, 1, &compute_prepare, 0, nullptr);
    }*/
}

static void issueShadowGen(Device &dev,
                           Frame &frame,
                           Pipeline<1> &pipeline,
                           VkCommandBuffer draw_cmd,
                           uint32_t view_idx,
                           uint32_t world_idx,
                           uint32_t max_views)
{
    {
        VkBufferMemoryBarrier compute_prepare = {
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT,
            VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            frame.renderInput.buffer,
            0, 
            (VkDeviceSize)frame.renderInputSize
        };

        dev.dt.cmdPipelineBarrier(draw_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 
                0, 0, nullptr, 1, &compute_prepare, 0, nullptr);
    }


    dev.dt.cmdBindPipeline(draw_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.hdls[0]);

    render::shader::ShadowGenPushConst push_const = { view_idx, world_idx };

    dev.dt.cmdPushConstants(draw_cmd,
                            pipeline.layout,
                            VK_SHADER_STAGE_COMPUTE_BIT,
                            0, sizeof(render::shader::ShadowGenPushConst),
                            &push_const);

    dev.dt.cmdBindDescriptorSets(draw_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.layout, 0, 1, &frame.shadowGenSet, 0, nullptr);

    uint32_t num_workgroups_x = utils::divideRoundUp(max_views, 32_u32);
    dev.dt.cmdDispatch(draw_cmd, num_workgroups_x, 1, 1);


    {
        VkBufferMemoryBarrier compute_prepare = {
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT,
            VK_ACCESS_MEMORY_WRITE_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            frame.renderInput.buffer,
            0, 
            (VkDeviceSize)frame.renderInputSize
        };

        dev.dt.cmdPipelineBarrier(draw_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                0, 0, nullptr, 1, &compute_prepare, 0, nullptr);
    }
}

static ViewerRendererState initState(RenderContext &rctx,
                                     const RenderWindow *window)
{
    Backend &backend = rctx.backend;
    Device &dev = rctx.dev;

    PresentationState present(
        backend, dev, window, InternalConfig::numFrames, true);

    ImGuiRenderState imgui_state = imguiInit(
        window->hdl, dev, backend, rctx.renderQueue,
        rctx.pipelineCache, VK_FORMAT_R8G8B8A8_UNORM,
        InternalConfig::depthFormat);

    Pipeline<1> object_shadow_draw = makeShadowDrawPipeline(
        dev, rctx.pipelineCache, rctx.shadowPass,
        rctx.repeatSampler, rctx.clampSampler, InternalConfig::numFrames);
    Pipeline<1> deferred_lighting = makeDeferredLightingPipeline(
        dev, rctx.pipelineCache, rctx.clampSampler, InternalConfig::numFrames);
    Pipeline<1> shadow_gen = makeShadowGenPipeline(
        dev, rctx.pipelineCache, rctx.clampSampler, InternalConfig::numFrames);
    Pipeline<1> blur = makeBlurPipeline(
        dev, rctx.pipelineCache, rctx.clampSampler, InternalConfig::numFrames);

    //Voxelizer Changes
    Pipeline<1> voxel_mesh_gen = makeVoxelMeshGenPipeline(
        dev, rctx.pipelineCache, InternalConfig::numFrames);
    Pipeline<1> voxel_draw = makeVoxelDrawPipeline(dev, rctx.pipelineCache,
        rctx.renderPass, rctx.repeatSampler, rctx.clampSampler,
        InternalConfig::numFrames);

    Pipeline<1> quad_draw = makeQuadPipeline(dev, rctx.pipelineCache,
        rctx.clampSampler, InternalConfig::numFrames,
        imgui_state.renderPass);

    Pipeline<1> grid_draw = makeGridDrawPipeline(dev, rctx.pipelineCache,
        InternalConfig::numFrames);
    
    HeapArray<Frame> frames(InternalConfig::numFrames);

    for (CountT i = 0; i < frames.size(); i++) {
        makeFrame(&frames[i],
                  dev, rctx.alloc,
                  window->width, window->height,
                  rctx.engine_interop_.maxViewsPerWorld,
                  rctx.engine_interop_.maxInstancesPerWorld,
                  rctx.voxel_config_,
                  rctx.renderPass,
                  imgui_state.renderPass,
                  rctx.shadowPass,
                  rctx.instanceCull.descPool.makeSet(),
                  rctx.objectDraw.descPool.makeSet(),
                  deferred_lighting.descPool.makeSet(),
                  shadow_gen.descPool.makeSet(),
                  blur.descPool.makeSet(),
                  voxel_mesh_gen.descPool.makeSet(),
                  voxel_draw.descPool.makeSet(),
                  quad_draw.descPool.makeSet(),
                  grid_draw.descPool.makeSet(),
                  rctx.sky_,
                  rctx.batchRenderer->getImportedBuffers(0),
                  rctx.batchRenderer->getRGBBuffer(),
                  rctx.batchRenderer->getDepthBuffer());
    }

    HostBuffer screenshot_buffer = rctx.alloc.makeStagingBuffer(
        frames[0].fb.colorAttachment.reqs.size);

    return ViewerRendererState {
        .rctx = rctx,
        .dev = dev,
        .window = window,
        .present = std::move(present),
        .presentWrapper { rctx.renderQueue, false },
        .imguiState = std::move(imgui_state),
        .fbImguiClear = makeImguiClearValues(),
        .fbWidth = window->width,
        .fbHeight = window->height,
        .fbClear = makeClearValues(),
        .fbShadowClear = makeShadowClearValues(),
        .objectShadowDraw = std::move(object_shadow_draw),
        .deferredLighting = std::move(deferred_lighting),
        .shadowGen = std::move(shadow_gen),
        .blur = std::move(blur),
        .voxelMeshGen = std::move(voxel_mesh_gen),
        .voxelDraw = std::move(voxel_draw),
        .quadDraw = std::move(quad_draw),
        .gridDraw = std::move(grid_draw),
        .curFrame = 0,
        .frames = std::move(frames),
        .globalFrameNum = 0,
        .screenshotBuffer = std::move(screenshot_buffer),
    };
}

bool ViewerRendererState::renderGridFrame(const viz::ViewerControl &viz_ctrl)
{
    Frame &frame = frames[curFrame];

    VkCommandBuffer draw_cmd = frame.drawCmd;
    { // Get command buffer for this frame and start it
        REQ_VK(dev.dt.resetCommandPool(dev.hdl, frame.drawCmdPool, 0));
        VkCommandBufferBeginInfo begin_info {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        REQ_VK(dev.dt.beginCommandBuffer(draw_cmd, &begin_info));
    }

    { // Pack the view for the fly camera and copy it to the render input
        packView(dev, frame.viewStaging, viz_ctrl.flyCam, fbWidth, fbHeight);
        VkBufferCopy view_copy {
            .srcOffset = 0,
            .dstOffset = frame.cameraViewOffset,
            .size = sizeof(PackedViewData)
        };
        dev.dt.cmdCopyBuffer(draw_cmd, frame.viewStaging.buffer,
                             frame.renderInput.buffer,
                             1, &view_copy);
    }

    { // Pack the lighting data and copy it to the render input
        packLighting(dev, frame.lightStaging, rctx.lights_);
        VkBufferCopy light_copy {
            .srcOffset = 0,
            .dstOffset = frame.lightOffset,
            .size = sizeof(DirectionalLight) * InternalConfig::maxLights
        };
        dev.dt.cmdCopyBuffer(draw_cmd, frame.lightStaging.buffer,
                             frame.renderInput.buffer,
                             1, &light_copy);
    }

    { // Pack the sky data and copy it to the render input
        packSky(dev, frame.skyStaging);
        VkBufferCopy sky_copy {
            .srcOffset = 0,
            .dstOffset = frame.skyOffset,
            .size = sizeof(SkyData)
        };
        dev.dt.cmdCopyBuffer(draw_cmd, frame.skyStaging.buffer,
                             frame.renderInput.buffer,
                             1, &sky_copy);
    }

    { // Issue grid drawing
        // issueLightingPass(dev, frame, deferred_lighting_, draw_cmd, cam, view_idx);
        issueGridDrawPass(rctx, frame, gridDraw, draw_cmd, viz_ctrl);
    }

    bool prepare_screenshot = viz_ctrl.requestedScreenshot ||
        (getenv("SCREENSHOT_PATH") && globalFrameNum == 0);

    if (prepare_screenshot) {
        array<VkImageMemoryBarrier, 1> prepare {{
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_TRANSFER_READ_BIT,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                frame.fb.colorAttachment.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            }
        }};

        VkBufferMemoryBarrier buffer_prepare = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_NONE,
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = screenshotBuffer.buffer,
            .offset = 0,
            .size = VK_WHOLE_SIZE
        };

        dev.dt.cmdPipelineBarrier(draw_cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0, nullptr, 1, &buffer_prepare,
                prepare.size(), prepare.data());

        VkBufferImageCopy region = {
            .bufferOffset = 0, .bufferRowLength = 0, .bufferImageHeight = 0,
            .imageSubresource = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1
            },
            .imageOffset = {},
            .imageExtent = {frame.fb.colorAttachment.width,
                frame.fb.colorAttachment.height, 1}
        };

        dev.dt.cmdCopyImageToBuffer(draw_cmd, frame.fb.colorAttachment.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                    screenshotBuffer.buffer, 1, &region);

        prepare[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        prepare[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        prepare[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        prepare[0].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        dev.dt.cmdPipelineBarrier(draw_cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0, nullptr, 0, nullptr,
                prepare.size(), prepare.data());
    }

    REQ_VK(dev.dt.endCommandBuffer(draw_cmd));

    uint32_t wait_count = 1;

    VkSemaphore waits[] = {
        rctx.batchRenderer->getLatestWaitSemaphore()
    };

    VkPipelineStageFlags wait_flags[] = {
        VK_PIPELINE_STAGE_TRANSFER_BIT
    };

    if (waits[0] == VK_NULL_HANDLE) {
        wait_count = 0;
    }

    if (!rctx.batchRenderer->didRender) {
        wait_count = 0;
    } else {
        rctx.batchRenderer->didRender = 0;
    }

    VkSubmitInfo gfx_submit {
        VK_STRUCTURE_TYPE_SUBMIT_INFO,
        nullptr,
        wait_count,
        waits,
        wait_flags,
        1,
        &draw_cmd,
        1,
        &frame.renderFinished,
    };

    REQ_VK(dev.dt.queueSubmit(rctx.renderQueue, 1, &gfx_submit,
                              VK_NULL_HANDLE));

    return prepare_screenshot;
}

bool ViewerRendererState::renderFlycamFrame(const ViewerControl &viz_ctrl)
{
    uint32_t view_idx = viz_ctrl.viewIdx;
    uint32_t world_idx = viz_ctrl.worldIdx;

    Frame &frame = frames[curFrame];

    VkCommandBuffer draw_cmd = frame.drawCmd;
    { // Get command buffer for this frame and start it
        REQ_VK(dev.dt.resetCommandPool(dev.hdl, frame.drawCmdPool, 0));
        VkCommandBufferBeginInfo begin_info {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        REQ_VK(dev.dt.beginCommandBuffer(draw_cmd, &begin_info));
    }

    { // Pack the view for the fly camera and copy it to the render input
        packView(dev, frame.viewStaging,
                 viz_ctrl.flyCam, fbWidth, fbHeight);
        VkBufferCopy view_copy {
            .srcOffset = 0,
            .dstOffset = frame.cameraViewOffset,
            .size = sizeof(PackedViewData)
        };
        dev.dt.cmdCopyBuffer(draw_cmd, frame.viewStaging.buffer,
                             frame.renderInput.buffer,
                             1, &view_copy);
    }

    { // Pack the lighting data and copy it to the render input
        packLighting(dev, frame.lightStaging, rctx.lights_);
        VkBufferCopy light_copy {
            .srcOffset = 0,
            .dstOffset = frame.lightOffset,
            .size = sizeof(DirectionalLight) * InternalConfig::maxLights
        };
        dev.dt.cmdCopyBuffer(draw_cmd, frame.lightStaging.buffer,
                             frame.renderInput.buffer,
                             1, &light_copy);
    }

    { // Pack the sky data and copy it to the render input
        packSky(dev, frame.skyStaging);
        VkBufferCopy sky_copy {
            .srcOffset = 0,
            .dstOffset = frame.skyOffset,
            .size = sizeof(SkyData)
        };
        dev.dt.cmdCopyBuffer(draw_cmd, frame.skyStaging.buffer,
                             frame.renderInput.buffer,
                             1, &sky_copy);
    }

    { // Reset the draw count to zero
        dev.dt.cmdFillBuffer(draw_cmd, frame.renderInput.buffer,
                             frame.drawCountOffset, sizeof(uint32_t), 0);
    }

    { // Issue shadow pass
        issueShadowGen(dev, frame, shadowGen, draw_cmd,
                       view_idx, world_idx, rctx.engine_interop_.maxViewsPerWorld);
    }

    const uint32_t num_voxels = rctx.voxel_config_.xLength * 
        rctx.voxel_config_.yLength *
        rctx.voxel_config_.zLength;

    { // Issue the voxel generation compute shader if needed
        if (num_voxels > 0) {
            dev.dt.cmdFillBuffer(draw_cmd, frame.voxelVBO.buffer,
                                 0, sizeof(float) * num_voxels * 6 * 4 * 8, 0);

            VkBufferCopy voxel_copy = {
                .srcOffset = 0,
                .dstOffset = 0,
                .size = num_voxels * sizeof(int32_t),
            };

            dev.dt.cmdCopyBuffer(draw_cmd,
                                 rctx.engine_interop_.voxelHdl,
                                 frame.voxelData.buffer,
                                 1, &voxel_copy);

            issueVoxelGen(dev, frame, 
                          voxelMeshGen, 
                          draw_cmd,
                          view_idx,
                          rctx.engine_interop_.maxInstancesPerWorld,
                          rctx.voxel_config_);
        }
    }

    uint32_t cur_num_instances = *rctx.engine_interop_.bridge.totalNumInstances;
    uint32_t cur_num_views = *rctx.engine_interop_.bridge.totalNumViews;

    { // Generate draw commands from the flycam.
        uint32_t draw_cmd_bytes = sizeof(VkDrawIndexedIndirectCommand) *
                                  rctx.engine_interop_.maxInstancesPerWorld * 10;
        dev.dt.cmdFillBuffer(draw_cmd, frame.renderInput.buffer,
                             frame.drawCmdOffset,
                             draw_cmd_bytes,
                             0);

        VkMemoryBarrier copy_barrier {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        };

        dev.dt.cmdPipelineBarrier(draw_cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 1, &copy_barrier, 0, nullptr, 0, nullptr);

        issueCulling(dev, draw_cmd, frame, rctx.instanceCull,
                     rctx.asset_set_cull_,
                     world_idx, cur_num_instances, cur_num_views,
                     rctx.num_worlds_);
    }

    { // Shadow pass
        dev.dt.cmdBindPipeline(draw_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                objectShadowDraw.hdls[0]);

        std::array draw_descriptors {
            frame.drawShaderSet,
            rctx.asset_set_draw_,
        };

        dev.dt.cmdBindDescriptorSets(draw_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                objectShadowDraw.layout, 0,
                draw_descriptors.size(),
                draw_descriptors.data(),
                0, nullptr);

        DrawPushConst draw_const {
            (uint32_t)view_idx,
            world_idx
        };

        dev.dt.cmdPushConstants(draw_cmd, objectShadowDraw.layout,
                VK_SHADER_STAGE_VERTEX_BIT |
                VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                sizeof(DrawPushConst), &draw_const);

        dev.dt.cmdBindIndexBuffer(draw_cmd, rctx.loaded_assets_[0].buf.buffer,
                rctx.loaded_assets_[0].idxBufferOffset,
                VK_INDEX_TYPE_UINT32);

        VkViewport viewport {
            0,
                0,
                (float)InternalConfig::shadowMapSize,
                (float)InternalConfig::shadowMapSize,
                0.f,
                1.f,
        };

        dev.dt.cmdSetViewport(draw_cmd, 0, 1, &viewport);

        VkRect2D scissor {
            { 0, 0 },
                { InternalConfig::shadowMapSize, InternalConfig::shadowMapSize },
        };

        dev.dt.cmdSetScissor(draw_cmd, 0, 1, &scissor);

        VkRenderPassBeginInfo render_pass_info;
        render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        render_pass_info.pNext = nullptr;
        render_pass_info.renderPass = rctx.shadowPass;
        render_pass_info.framebuffer = frame.shadowFB.hdl;
        render_pass_info.clearValueCount = fbShadowClear.size();
        render_pass_info.pClearValues = fbShadowClear.data();
        render_pass_info.renderArea.offset = {
            0, 0,
        };
        render_pass_info.renderArea.extent = {
            InternalConfig::shadowMapSize, InternalConfig::shadowMapSize,
        };

        dev.dt.cmdBeginRenderPass(draw_cmd, &render_pass_info,
                VK_SUBPASS_CONTENTS_INLINE);

        dev.dt.cmdDrawIndexedIndirect(draw_cmd,
                frame.renderInput.buffer,
                frame.drawCmdOffset,
                rctx.engine_interop_.maxInstancesPerWorld * 10,
                sizeof(DrawCmd));

        dev.dt.cmdEndRenderPass(draw_cmd);

        issueShadowBlurPass(dev, frame, blur, draw_cmd);
#if 1
        array<VkImageMemoryBarrier, 1> finish_prepare {{
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_MEMORY_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT,
                VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                frame.shadowFB.depthAttachment.image,
                {
                    VK_IMAGE_ASPECT_DEPTH_BIT,
                    0, 1, 0, 1
                },
            }
        }};

        dev.dt.cmdPipelineBarrier(draw_cmd,
                VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0, nullptr, 0, nullptr,
                finish_prepare.size(), finish_prepare.data());
#endif
    }

    dev.dt.cmdBindPipeline(draw_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                           rctx.objectDraw.hdls[0]);

    std::array draw_descriptors {
        frame.drawShaderSet,
        rctx.asset_set_draw_,
        rctx.asset_set_mat_tex_
    };

    dev.dt.cmdBindDescriptorSets(draw_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 rctx.objectDraw.layout, 0,
                                 draw_descriptors.size(),
                                 draw_descriptors.data(),
                                 0, nullptr);

    DrawPushConst draw_const {
        (uint32_t)view_idx,
        world_idx
    };

    dev.dt.cmdPushConstants(draw_cmd, rctx.objectDraw.layout,
                            VK_SHADER_STAGE_VERTEX_BIT |
                            VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                            sizeof(DrawPushConst), &draw_const);

    dev.dt.cmdBindIndexBuffer(draw_cmd, rctx.loaded_assets_[0].buf.buffer,
                              rctx.loaded_assets_[0].idxBufferOffset,
                              VK_INDEX_TYPE_UINT32);

    VkViewport viewport {
        0,
        0,
        (float)fbWidth,
        (float)fbHeight,
        0.f,
        1.f,
    };

    dev.dt.cmdSetViewport(draw_cmd, 0, 1, &viewport);

    VkRect2D scissor {
        { 0, 0 },
        { fbWidth, fbHeight },
    };

    dev.dt.cmdSetScissor(draw_cmd, 0, 1, &scissor);


    VkRenderPassBeginInfo render_pass_info;
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass_info.pNext = nullptr;
    render_pass_info.renderPass = rctx.renderPass;
    render_pass_info.framebuffer = frame.fb.hdl;
    render_pass_info.clearValueCount = fbClear.size();
    render_pass_info.pClearValues = fbClear.data();
    render_pass_info.renderArea.offset = {
        0, 0,
    };
    render_pass_info.renderArea.extent = {
        fbWidth, fbHeight,
    };

    dev.dt.cmdBeginRenderPass(draw_cmd, &render_pass_info,
                              VK_SUBPASS_CONTENTS_INLINE);

    dev.dt.cmdDrawIndexedIndirect(draw_cmd,
                                  frame.renderInput.buffer,
                                  frame.drawCmdOffset,
                                  rctx.engine_interop_.maxInstancesPerWorld * 10,
                                  sizeof(DrawCmd));

    if (num_voxels > 0) {
        dev.dt.cmdBindPipeline(draw_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            voxelDraw.hdls[0]);

        std::array voxel_draw_descriptors {
            frame.voxelDrawSet,
            rctx.asset_set_mat_tex_
        };

        dev.dt.cmdBindDescriptorSets(draw_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            voxelDraw.layout, 0,
            voxel_draw_descriptors.size(),
            voxel_draw_descriptors.data(),
            0, nullptr);

        DrawPushConst voxel_draw_const{
            (uint32_t)view_idx,
            world_idx
        };

        dev.dt.cmdPushConstants(draw_cmd, voxelDraw.layout,
            VK_SHADER_STAGE_VERTEX_BIT |
            VK_SHADER_STAGE_FRAGMENT_BIT, 0,
            sizeof(DrawPushConst), &voxel_draw_const);

        dev.dt.cmdBindIndexBuffer(draw_cmd, frame.voxelIndexBuffer.buffer,
            0,
            VK_INDEX_TYPE_UINT32);
        dev.dt.cmdDrawIndexed(draw_cmd, static_cast<uint32_t>(num_voxels * 6 * 6),
            1, 0, 0, 0);
    }

    dev.dt.cmdEndRenderPass(draw_cmd);

    issueLightingPass(dev, frame, deferredLighting, draw_cmd, viz_ctrl.flyCam,
                      view_idx, world_idx);

    bool prepare_screenshot = viz_ctrl.requestedScreenshot ||
        (getenv("SCREENSHOT_PATH") && globalFrameNum == 0);

    if (prepare_screenshot) {
        array<VkImageMemoryBarrier, 1> prepare {{
            {
                VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                nullptr,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                VK_ACCESS_TRANSFER_READ_BIT,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_QUEUE_FAMILY_IGNORED,
                VK_QUEUE_FAMILY_IGNORED,
                frame.fb.colorAttachment.image,
                {
                    VK_IMAGE_ASPECT_COLOR_BIT,
                    0, 1, 0, 1
                },
            }
        }};

        VkBufferMemoryBarrier buffer_prepare = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_NONE,
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = screenshotBuffer.buffer,
            .offset = 0,
            .size = VK_WHOLE_SIZE
        };

        dev.dt.cmdPipelineBarrier(draw_cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0, nullptr, 1, &buffer_prepare,
                prepare.size(), prepare.data());

        VkBufferImageCopy region = {
            .bufferOffset = 0, .bufferRowLength = 0, .bufferImageHeight = 0,
            .imageSubresource = {
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1
            },
            .imageOffset = {},
            .imageExtent = {frame.fb.colorAttachment.width, frame.fb.colorAttachment.height, 1}
        };

        dev.dt.cmdCopyImageToBuffer(draw_cmd, frame.fb.colorAttachment.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                    screenshotBuffer.buffer, 1, &region);

        prepare[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        prepare[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        prepare[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        prepare[0].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        dev.dt.cmdPipelineBarrier(draw_cmd,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0, nullptr, 0, nullptr,
                prepare.size(), prepare.data());
    }

    REQ_VK(dev.dt.endCommandBuffer(draw_cmd));

    uint32_t wait_count = 1;

    VkSemaphore waits[] = {
        rctx.batchRenderer->getLatestWaitSemaphore()
    };

    VkPipelineStageFlags wait_flags[] = {
        VK_PIPELINE_STAGE_TRANSFER_BIT
    };

    if (waits[0] == VK_NULL_HANDLE) {
        wait_count = 0;
    }

    if (!rctx.batchRenderer->didRender) {
        wait_count = 0;
    } else {
        rctx.batchRenderer->didRender = 0;
    }

    VkSubmitInfo gfx_submit {
        VK_STRUCTURE_TYPE_SUBMIT_INFO,
        nullptr,
        wait_count,
        waits,
        wait_flags,
        1,
        &draw_cmd,
        1,
        &frame.renderFinished,
    };

    REQ_VK(dev.dt.queueSubmit(rctx.renderQueue, 1, &gfx_submit,
                              VK_NULL_HANDLE));

    return prepare_screenshot;
}

bool ViewerRendererState::renderGUIAndPresent(
    const viz::ViewerControl &viz_ctrl,
    bool prepare_screenshot)
{
    Frame &frame = frames[curFrame];

    bool need_resize;
    currentSwapchainIndex = present.acquireNext(dev, 
            frame.swapchainReady, need_resize);

    if (need_resize) {
        handleResize();

        return false;
    }

    VkImage swapchain_img = present.getImage(currentSwapchainIndex);

    VkCommandBuffer draw_cmd = frame.presentCmd;
    { // Get command buffer for this frame and start it
        REQ_VK(dev.dt.resetCommandPool(dev.hdl, frame.presentCmdPool, 0));
        VkCommandBufferBeginInfo begin_info {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        REQ_VK(dev.dt.beginCommandBuffer(draw_cmd, &begin_info));
    }

    VkRenderPassBeginInfo render_pass_info;
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass_info.pNext = nullptr;
    render_pass_info.renderPass = rctx.renderPass;
    render_pass_info.framebuffer = frame.fb.hdl;
    render_pass_info.clearValueCount = fbClear.size();
    render_pass_info.pClearValues = fbClear.data();
    render_pass_info.renderArea.offset = {
        0, 0,
    };

    render_pass_info.renderArea.extent = {
        fbWidth, fbHeight,
    };

    render_pass_info.framebuffer = frame.imguiFB.hdl;
    render_pass_info.renderPass = imguiState.renderPass;
    dev.dt.cmdBeginRenderPass(draw_cmd, &render_pass_info,
                              VK_SUBPASS_CONTENTS_INLINE);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), draw_cmd);

#ifdef ENABLE_BATCH_RENDERER
    { // Draw the quads
        // issueQuadDraw(dev, draw_cmd, frame, *quad_draw_);
    }
#endif

    dev.dt.cmdEndRenderPass(draw_cmd);

    array<VkImageMemoryBarrier, 2> blit_prepare {{
        {
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            nullptr,
            0,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            swapchain_img,
            {
                VK_IMAGE_ASPECT_COLOR_BIT,
                0, 1, 0, 1
            },
        },
        {
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            frame.fb.colorAttachment.image,
            {
                VK_IMAGE_ASPECT_COLOR_BIT,
                0, 1, 0, 1
            },
        }
    }};

    dev.dt.cmdPipelineBarrier(draw_cmd,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT |
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0, nullptr, 0, nullptr,
        blit_prepare.size(), blit_prepare.data());

    VkImageBlit blit_region {
        { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        {
            { 0, 0, 0 }, 
            { (int32_t)fbWidth, (int32_t)fbHeight, 1 },
        },
        { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        {
            { 0, 0, 0 }, 
            { (int32_t)fbWidth, (int32_t)fbHeight, 1 },
        },
    };

    dev.dt.cmdBlitImage(draw_cmd,
                        frame.fb.colorAttachment.image,
                        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        swapchain_img,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        1, &blit_region,
                        VK_FILTER_NEAREST);

    VkImageMemoryBarrier swapchain_prepare {
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        nullptr,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        0,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED,
        swapchain_img,
        {
            VK_IMAGE_ASPECT_COLOR_BIT,
            0, 1, 0, 1
        },
    };

    dev.dt.cmdPipelineBarrier(draw_cmd,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                              0,
                              0, nullptr, 0, nullptr,
                              1, &swapchain_prepare);

    REQ_VK(dev.dt.endCommandBuffer(draw_cmd));

    VkSemaphore waits[] = {
        frame.swapchainReady, frame.renderFinished
    };

    VkPipelineStageFlags wait_flags[] = {
        (VkPipelineStageFlags)VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 
        (VkPipelineStageFlags)(
            prepare_screenshot ? VK_PIPELINE_STAGE_TRANSFER_BIT :
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
        ),
    };

    VkSubmitInfo gfx_submit {
        VK_STRUCTURE_TYPE_SUBMIT_INFO,
        nullptr,
        2,
        waits,
        wait_flags,
        1,
        &draw_cmd,
        1,
        &frame.guiRenderFinished,
    };

    REQ_VK(dev.dt.resetFences(dev.hdl, 1, &frame.cpuFinished));
    REQ_VK(dev.dt.queueSubmit(rctx.renderQueue, 1, &gfx_submit,
                              frame.cpuFinished));

    present.present(dev, currentSwapchainIndex, presentWrapper,
                    1, &frame.guiRenderFinished,
                    need_resize);

    if (need_resize) {
        handleResize();

        return false;
    }

    curFrame = (curFrame + 1) % frames.size();

    if (prepare_screenshot) {
        const char *ss_path = viz_ctrl.screenshotFilePath;

        if (!viz_ctrl.requestedScreenshot) {
            ss_path = getenv("SCREENSHOT_PATH");
        }

        dev.dt.deviceWaitIdle(dev.hdl);

        void *pixels = screenshotBuffer.ptr;

        std::string dst_file = ss_path;
        int ret = stbi_write_bmp(
            dst_file.c_str(), frame.fb.colorAttachment.width,
            frame.fb.colorAttachment.height, 4, pixels);

        if (ret) {
            printf("Wrote %s\n", dst_file.c_str());
        }
    }

    globalFrameNum += 1;

    return true;
}

void ViewerRendererState::destroy()
{
    rctx.waitForIdle();

    for (Frame &f : frames) {
        dev.dt.destroySemaphore(dev.hdl, f.swapchainReady, nullptr);
        dev.dt.destroySemaphore(dev.hdl, f.renderFinished, nullptr);
        dev.dt.destroySemaphore(dev.hdl, f.guiRenderFinished, nullptr);

        dev.dt.destroyFence(dev.hdl, f.cpuFinished, nullptr);
        dev.dt.destroyCommandPool(dev.hdl, f.drawCmdPool, nullptr);
        dev.dt.destroyCommandPool(dev.hdl, f.presentCmdPool, nullptr);

        dev.dt.destroyFramebuffer(dev.hdl, f.fb.hdl, nullptr);

        dev.dt.destroyFramebuffer(dev.hdl, f.imguiFB.hdl, nullptr);

        dev.dt.destroyFramebuffer(dev.hdl, f.shadowFB.hdl, nullptr);

        dev.dt.destroyImageView(dev.hdl, f.fb.colorView, nullptr);
        dev.dt.destroyImageView(dev.hdl, f.fb.normalView, nullptr);
        dev.dt.destroyImageView(dev.hdl, f.fb.positionView, nullptr);
        dev.dt.destroyImageView(dev.hdl, f.fb.depthView, nullptr);

        dev.dt.destroyImageView(dev.hdl, f.shadowFB.varianceView, nullptr);
        dev.dt.destroyImageView(dev.hdl, f.shadowFB.depthView, nullptr);

        dev.dt.destroyImageView(dev.hdl, f.shadowFB.intermediateView, nullptr);
    }

    dev.dt.destroyPipeline(dev.hdl, gridDraw.hdls[0], nullptr);
    dev.dt.destroyPipelineLayout(dev.hdl, gridDraw.layout, nullptr);

    dev.dt.destroyPipeline(dev.hdl, quadDraw.hdls[0], nullptr);
    dev.dt.destroyPipelineLayout(dev.hdl, quadDraw.layout, nullptr);

    dev.dt.destroyPipeline(dev.hdl, voxelDraw.hdls[0], nullptr);
    dev.dt.destroyPipelineLayout(dev.hdl, voxelDraw.layout, nullptr);

    dev.dt.destroyPipeline(dev.hdl, voxelMeshGen.hdls[0], nullptr);
    dev.dt.destroyPipelineLayout(dev.hdl, voxelMeshGen.layout, nullptr);

    dev.dt.destroyPipeline(dev.hdl, blur.hdls[0], nullptr);
    dev.dt.destroyPipelineLayout(dev.hdl, blur.layout, nullptr);

    dev.dt.destroyPipeline(dev.hdl, shadowGen.hdls[0], nullptr);
    dev.dt.destroyPipelineLayout(dev.hdl, shadowGen.layout, nullptr);

    dev.dt.destroyPipeline(dev.hdl, deferredLighting.hdls[0], nullptr);
    dev.dt.destroyPipelineLayout(dev.hdl, deferredLighting.layout, nullptr);

    dev.dt.destroyPipeline(dev.hdl, objectShadowDraw.hdls[0], nullptr);
    dev.dt.destroyPipelineLayout(dev.hdl, objectShadowDraw.layout, nullptr);

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();

    dev.dt.destroyRenderPass(dev.hdl, imguiState.renderPass, nullptr);
    dev.dt.destroyDescriptorPool(dev.hdl, imguiState.descPool, nullptr);

    present.destroy(dev);
}

void ViewerRendererState::handleResize()
{
    // Wait for everything to finish
    dev.dt.deviceWaitIdle(dev.hdl);

    uint32_t new_width = window->width;
    uint32_t new_height = window->height;


    // Recreate the swapchain first
    present.resize(rctx.backend,
                   dev,
                   window,
                   InternalConfig::numFrames,
                   true);

    // Destroy the image views for the framebuffer
    for (uint32_t i = 0; i < InternalConfig::numFrames; ++i) {
        Frame &frame = frames[i];

        destroyFramebuffers(dev, rctx.alloc, 
                frame.fb, frame.imguiFB);

        auto [fb, imgui_fb] = makeFramebuffers(
                dev, rctx.alloc, new_width, new_height, 
                rctx.renderPass, imguiState.renderPass);

        new(&frame.fb) Framebuffer(std::move(fb));
        new(&frame.imguiFB) Framebuffer(std::move(imgui_fb));

        // Need to update relevant descriptor sets now (nightmare)
        std::array<VkWriteDescriptorSet, 16> desc_updates;
        uint32_t desc_counter = 0;

        VkDescriptorImageInfo gbuffer_albedo_info;
        gbuffer_albedo_info.imageView = fb.colorView;
        gbuffer_albedo_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        gbuffer_albedo_info.sampler = VK_NULL_HANDLE;

        DescHelper::storageImage(desc_updates[desc_counter++],
                frame.lightingSet, &gbuffer_albedo_info, 0);

        if (frame.gridDrawSet != VK_NULL_HANDLE) {
            DescHelper::storageImage(desc_updates[desc_counter++],
                    frame.gridDrawSet, &gbuffer_albedo_info, 0);
        }

        VkDescriptorImageInfo gbuffer_normal_info;
        gbuffer_normal_info.imageView = fb.normalView;
        gbuffer_normal_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        gbuffer_normal_info.sampler = VK_NULL_HANDLE;

        DescHelper::storageImage(desc_updates[desc_counter++],
                frame.lightingSet, &gbuffer_normal_info, 1);

        VkDescriptorImageInfo gbuffer_position_info;
        gbuffer_position_info.imageView = fb.positionView;
        gbuffer_position_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        gbuffer_position_info.sampler = VK_NULL_HANDLE;

        DescHelper::storageImage(desc_updates[desc_counter++],
                frame.lightingSet, &gbuffer_position_info, 2);

        DescHelper::update(dev, desc_updates.data(), desc_counter);

        dev.dt.destroyFence(dev.hdl, frame.cpuFinished, nullptr);
        dev.dt.destroySemaphore(dev.hdl, frame.renderFinished, nullptr);
        dev.dt.destroySemaphore(dev.hdl, frame.guiRenderFinished, nullptr);
        dev.dt.destroySemaphore(dev.hdl, frame.swapchainReady, nullptr);

        frame.cpuFinished = makeFence(dev, true);
        frame.renderFinished = makeBinarySemaphore(dev);
        frame.guiRenderFinished = makeBinarySemaphore(dev);
        frame.swapchainReady = makeBinarySemaphore(dev);
    }

#if 0
    if (rctx.batchRenderer) {
        rctx.batchRenderer->recreateSemaphores();
    }
#endif

    fbWidth = new_width;
    fbHeight = new_height;

    const_cast<vk::RenderWindow *>(window)->needResize = false;
}

void ViewerRendererState::recreateSemaphores()
{
    dev.dt.deviceWaitIdle(dev.hdl);

    for (uint32_t i = 0; i < InternalConfig::numFrames; ++i) {
        Frame &frame = frames[i];

        dev.dt.destroyFence(dev.hdl, frame.cpuFinished, nullptr);
        dev.dt.destroySemaphore(dev.hdl, frame.renderFinished, nullptr);
        dev.dt.destroySemaphore(dev.hdl, frame.guiRenderFinished, nullptr);
        dev.dt.destroySemaphore(dev.hdl, frame.swapchainReady, nullptr);

        frame.cpuFinished = makeFence(dev, true);
        frame.renderFinished = makeBinarySemaphore(dev);
        frame.guiRenderFinished = makeBinarySemaphore(dev);
        frame.swapchainReady = makeBinarySemaphore(dev);
    }

#if 0
    if (rctx.batchRenderer) {
        rctx.batchRenderer->recreateSemaphores();
    }
#endif
}

ViewerRenderer::ViewerRenderer(const render::RenderManager &render_mgr,
                               const Window *window)
    : state_(initState(render_mgr.renderContext(),
                       static_cast<const RenderWindow *>(window)))
{}

ViewerRenderer::~ViewerRenderer()
{
    state_.destroy();
}

void ViewerRenderer::waitUntilFrameReady()
{
    Frame &frame = state_.frames[state_.curFrame];
    // Wait until frame using this slot has finished

    REQ_VK(state_.dev.dt.waitForFences(
        state_.dev.hdl, 1, &frame.cpuFinished, VK_TRUE, UINT64_MAX));
}

void ViewerRenderer::waitForIdle()
{
    state_.rctx.waitForIdle();
}

void ViewerRenderer::startFrame()
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
}

void ViewerRenderer::render(const viz::ViewerControl &viz_ctrl)
{
    bool prepare_screenshot = 0;

    if (viz_ctrl.viewerType == viz::ViewerType::Flycam) {
        prepare_screenshot = state_.renderFlycamFrame(viz_ctrl);
    } else if (viz_ctrl.viewerType == viz::ViewerType::Grid) {
        prepare_screenshot = state_.renderGridFrame(viz_ctrl);
    }

    state_.renderGUIAndPresent(viz_ctrl, prepare_screenshot);
}


CountT ViewerRenderer::loadObjects(Span<const imp::SourceObject> objs,
                                   Span<const imp::SourceMaterial> mats,
                                   Span<const imp::SourceTexture> textures)
{
    return state_.rctx.loadObjects(objs, mats, textures);
}

void ViewerRenderer::configureLighting(Span<const render::LightConfig> lights)
{
    state_.rctx.configureLighting(lights);
}

bool ViewerRenderer::needResize() const
{
    // If the width/height doesn't match, that means a resize is needed
    return state_.window->width != state_.fbWidth ||
        state_.window->height != state_.fbHeight ||
        state_.window->needResize;
}

void ViewerRenderer::handleResize()
{
    state_.handleResize();
    // const_cast<vk::RenderWindow *>(state_.window)->needResize = true;
}

}
