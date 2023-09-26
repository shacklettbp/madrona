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

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <madrona/cuda_utils.hpp>

namespace madrona {
namespace render {

using namespace vk;

template <size_t N>
struct Pipeline {
    render::vk::PipelineShaders shaders;
    VkPipelineLayout layout;
    std::array<VkPipeline, N> hdls;
    render::vk::FixedDescriptorPool descPool;
};

static Backend makeBackend();
static VkQueue makeGFXQueue(const Device &dev, uint32_t idx);
static VkPipelineCache getPipelineCache(const Device &dev);
static Pipeline<1> makeComputePipeline(const Device &dev,
                                       VkPipelineCache pipeline_cache,
                                       uint32_t push_constant_size,
                                       uint32_t num_frames,
                                       const char *shader_file,
                                       const char *func_name = "main");

// GPU resources required for batch rendering for each frame in flight.
struct FrameResources {
    // There are total_num_views / max_layers_per_image elements in
    // these vectors.
    std::vector<LocalImage> layeredImages;
    std::vector<VkImageView> layeredImageViews;
};

struct BatchRenderer::Impl {
    Backend backend;
    Device dev;
    MemoryAllocator mem;
    VkQueue renderQueue;
    VkPipelineCache pipelineCache;

    // Pipeline which will prepare all the views (disperse the draw instance
    // information into separate buffers which will get fed into an indirect 
    // draw command).
    Pipeline<1> prepareViewsPipeline;

    Impl(uint32_t gpu_id, uint32_t num_frames_in_flight)
        : backend(makeBackend()),
          dev(backend.initDevice(gpu_id)),
          mem(dev, backend),
          renderQueue(makeGFXQueue(dev, 0)),
          pipelineCache(getPipelineCache(dev)),
          prepareViewsPipeline(makeComputePipeline(dev, pipelineCache,
                                                   0,
                                                   num_frames_in_flight,
                                                   "prepare_views.hlsl"))
    {
    }
};

BatchRenderer BatchRenderer::make(const Config &cfg)
{
    return {};
}

BatchRenderer::~BatchRenderer()
{
}

BatchRendererECSBridge *BatchRenderer::makeECSBridge()
{
    return (BatchRendererECSBridge *)
        cu::allocReadback(sizeof(BatchRendererECSBridge));
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

static Backend makeBackend()
{
    auto get_inst_addr = glfwGetInstanceProcAddress(
        VK_NULL_HANDLE, "vkGetInstanceProcAddr");

    bool enable_validation;
    char *validate_env = getenv("MADRONA_RENDER_VALIDATE");
    if (!validate_env || validate_env[0] == '0') {
        enable_validation = false;
    } else {
        enable_validation = true;
    }

    // Just to eliminate potential errors, keeping as much as possible the same 
    // code as in viewer_renderer.cpp
    uint32_t count;
    const char **extensions_ptr = glfwGetRequiredInstanceExtensions(&count);
    Span<const char *> extensions (extensions_ptr, (CountT)count);

    return Backend((void (*)())get_inst_addr, enable_validation, false,
                   extensions);
}

static VkQueue makeGFXQueue(const Device &dev, uint32_t idx)
{
    if (idx >= dev.numGraphicsQueues) {
        FATAL("Not enough graphics queues");
    }

    return makeQueue(dev, dev.gfxQF, idx);
}

static PipelineShaders makeShaders(const Device &dev,
                                   const char *shader_file,
                                   const char *func_name = "main")
{
    std::filesystem::path shader_dir =
        std::filesystem::path(STRINGIFY(SHADER_DIR));

    ShaderCompiler compiler;
    SPIRVShader spirv = compiler.compileHLSLFileToSPV(
        (shader_dir / shader_file).string().c_str(), {},
        {}, {func_name, ShaderStage::Compute });

    StackAlloc tmp_alloc;
    return PipelineShaders(dev, tmp_alloc,
                           Span<const SPIRVShader>(&spirv, 1), {});
}

static Pipeline<1> makeComputePipeline(const Device &dev,
                                       VkPipelineCache pipeline_cache,
                                       uint32_t push_constant_size,
                                       uint32_t num_frames,
                                       const char *shader_file,
                                       const char *func_name)
{
    PipelineShaders shader = makeShaders(dev, shader_file);

    VkPushConstantRange push_const = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = push_constant_size
    };

    std::vector<VkDescriptorSetLayout> desc_layouts;
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

    FixedDescriptorPool desc_pool(dev, shader, 0, num_frames);

    return Pipeline<1> {
        std::move(shader),
        layout,
        pipelines,
        std::move(desc_pool),
    };
}

}
}
