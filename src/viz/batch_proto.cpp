#include "batch_proto.hpp"
#include "viewer_renderer.hpp"
#include "shader.hpp"

#include "madrona/heap_array.hpp"

#include <filesystem>

#include "shader.hpp"
#include "vk/memory.hpp"


using namespace madrona::render;
using madrona::render::vk::checkVk;

namespace madrona::viz {

namespace consts {
inline constexpr uint32_t maxDrawsPerLayeredImage = 65536;
inline constexpr VkFormat colorFormat = VK_FORMAT_R8G8B8A8_UNORM;
inline constexpr VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;
inline constexpr uint32_t numDrawCmdBuffers = 3; // Triple buffering
}

////////////////////////////////////////////////////////////////////////////////
// LAYERED OUTPUT CREATION                                                    //
////////////////////////////////////////////////////////////////////////////////
struct ImportedBuffers {
    vk::LocalBuffer views;
    vk::LocalBuffer instances;
    vk::LocalBuffer instanceOffsets;
};





////////////////////////////////////////////////////////////////////////////////
// LAYERED OUTPUT CREATION                                                    //
////////////////////////////////////////////////////////////////////////////////

// A layered target will have a color image with the max amount of layers, depth
// image with max amount of layers.
struct LayeredTarget {
    vk::LocalImage color;
    vk::LocalImage depth;
};

static HeapArray<LayeredTarget> makeLayeredTargets(uint32_t width,
                                                   uint32_t height,
                                                   uint32_t max_num_views,
                                                   const vk::Device &dev,
                                                   vk::MemoryAllocator &alloc)
{
    uint32_t num_images = utils::divideRoundUp(max_num_views,
                                               dev.maxNumLayersPerImage);

    HeapArray<LayeredTarget> local_images (num_images);

    for (int i = 0; i < (int)num_images; ++i) {
        LayeredTarget target = {
            .color = alloc.makeColorAttachment(width, height,
                                                 dev.maxNumLayersPerImage,
                                                 consts::colorFormat),
            .depth = alloc.makeDepthAttachment(width, height,
                                                 dev.maxNumLayersPerImage,
                                                 consts::depthFormat)
        };

         
        local_images.emplace(i, std::move(target));
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
// GENERIC COMPUTE PIPELINE CREATION                                          //
////////////////////////////////////////////////////////////////////////////////
static vk::PipelineShaders makeShaders(const vk::Device &dev,
                                       const char *shader_file,
                                       const char *func_name = "main")
{
    std::filesystem::path shader_dir =
        std::filesystem::path(STRINGIFY(VIEWER_DATA_DIR));

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
                                       uint32_t num_frames,
                                       const char *shader_file,
                                       const char *func_name = "main")
{
    vk::PipelineShaders shader = makeShaders(dev, shader_file);

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

    vk::FixedDescriptorPool desc_pool(dev, shader, 0, num_frames);

    return Pipeline<1> {
        std::move(shader),
        layout,
        pipelines,
        std::move(desc_pool),
    };
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
    HeapArray<LayeredTarget> targets;
    // We use anything from double, triple, or whatever we can buffering to save
    // on memory usage
    HeapArray<vk::LocalBuffer> bufferedDrawCmds;

    vk::SparseBuffer sparseBufferTest;

    // This pipeline prepares the draw commands in the buffered draw cmds buffer
    // Pipeline<1> prepareViews;

    Impl(const Config &cfg, vk::Device &dev, vk::MemoryAllocator &mem, 
         VkPipelineCache);
};

BatchRendererProto::Impl::Impl(const Config &cfg,
                               vk::Device &dev,
                               vk::MemoryAllocator &mem,
                               VkPipelineCache pipeline_cache)
    : dev(dev), mem(mem), pipelineCache(pipeline_cache),
      maxNumViews(cfg.numWorlds * cfg.maxViewsPerWorld),
      targets(makeLayeredTargets(cfg.renderWidth, cfg.renderHeight,
                                 maxNumViews, dev, mem)),
      bufferedDrawCmds(makeDrawCmdBuffers(consts::numDrawCmdBuffers, dev, mem)),
      sparseBufferTest(mem.makeSparseBuffer(1024*1024).value())
      // prepareViews(makeComputePipeline(dev, pipelineCache, 0, 1, "prepare_views"))
{
}

BatchRendererProto::BatchRendererProto(const Config &cfg,
                                       vk::Device &dev,
                                       vk::MemoryAllocator &mem,
                                       VkPipelineCache pipeline_cache)
    : impl(std::make_unique<Impl>(cfg, dev, mem, pipeline_cache))
{
}

BatchRendererProto::~BatchRendererProto()
{
}

}
