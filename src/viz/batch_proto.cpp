#include "batch_proto.hpp"
#include "madrona/viz/interop.hpp"
#include "viewer_renderer.hpp"
#include "shader.hpp"

#include "madrona/heap_array.hpp"

#include <filesystem>

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
    vk::PipelineShaders shader = makeShaders(dev, shader_file);

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
    //view, instance info,instance data
    VkDescriptorSet viewInstanceSet;
};

void makeBatchFrame(vk::Device& dev, BatchFrame* frame, render::vk::MemoryAllocator &alloc, uint32_t num_worlds, uint32_t num_instances, uint32_t num_views
    , VkDescriptorSet viewInstanceSet) {
    VkDeviceSize view_size = (num_worlds * num_views) * sizeof(PerspectiveCameraData);
    vk::LocalBuffer views = alloc.makeLocalBuffer(view_size).value();

    VkDeviceSize instance_size = (num_worlds * num_instances) * sizeof(InstanceData);
    vk::LocalBuffer instances = alloc.makeLocalBuffer(instance_size).value();

    VkDeviceSize instance_offset_size = (num_worlds) * sizeof(uint32_t);
    vk::LocalBuffer instance_offsets = alloc.makeLocalBuffer(instance_offset_size).value();

    //Descriptor sets
    std::array<VkWriteDescriptorSet, 3> desc_updates;

    VkDescriptorBufferInfo view_info;
    view_info.buffer = views.buffer;
    view_info.offset = 0;
    view_info.range = view_size;
    vk::DescHelper::storage(desc_updates[0], viewInstanceSet, &view_info, 0);

    VkDescriptorBufferInfo instance_info;
    instance_info.buffer = instances.buffer;
    instance_info.offset = 0;
    instance_info.range = instance_size;
    vk::DescHelper::storage(desc_updates[1], viewInstanceSet, &instance_info, 1);

    VkDescriptorBufferInfo offset_info;
    offset_info.buffer = instance_offsets.buffer;
    offset_info.offset = 0;
    offset_info.range = instance_offset_size;
    vk::DescHelper::storage(desc_updates[2], viewInstanceSet, &offset_info, 2);

    vk::DescHelper::update(dev, desc_updates.data(), desc_updates.size());
    new (frame) BatchFrame{
        {std::move(views),std::move(instances),std::move(instance_offsets)},
        viewInstanceSet
    };
}

struct ViewBatch {
    //Draw cmds and drawdata
    vk::LocalBuffer drawBuffer;
    VkDescriptorSet drawBufferSet;
};

void makeViewBatch(vk::Device& dev, ViewBatch* batch, render::vk::MemoryAllocator &alloc, VkDescriptorSet drawBufferSet) {
    //Make Draw Buffers

    int64_t buffer_offsets[1];
    int64_t buffer_sizes[2] = {
        (int64_t)sizeof(shader::DrawCmd) * consts::maxDrawsPerLayeredImage,
        (int64_t)sizeof(shader::DrawData) * consts::maxDrawsPerLayeredImage
    };

    int64_t num_draw_bytes = utils::computeBufferOffsets(
        buffer_sizes, buffer_offsets, 256);

    vk::LocalBuffer drawBuffer = alloc.makeLocalBuffer(num_draw_bytes).value();

    std::array<VkWriteDescriptorSet, 2> desc_updates;

    VkDescriptorBufferInfo draw_cmd_info;
    draw_cmd_info.buffer = drawBuffer.buffer;
    draw_cmd_info.offset = 0;
    draw_cmd_info.range = buffer_sizes[0];

    vk::DescHelper::storage(desc_updates[0], drawBufferSet, &draw_cmd_info, 0);

    VkDescriptorBufferInfo draw_data_info;
    draw_data_info.buffer = drawBuffer.buffer;
    draw_data_info.offset = buffer_offsets[0];
    draw_data_info.range = buffer_sizes[1];

    vk::DescHelper::storage(desc_updates[1], drawBufferSet, &draw_data_info, 1);

    vk::DescHelper::update(dev, desc_updates.data(), desc_updates.size());

    new (batch) ViewBatch{
        std::move(drawBuffer),
        drawBufferSet
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

    Pipeline<1> prepareViews;

    //One batch is num_layers views at once
    HeapArray<ViewBatch> viewBatches;

    //One frame is on simulation frame
    HeapArray<BatchFrame> batchFrames;

    VkDescriptorSet assetSet;

    // This pipeline prepares the draw commands in the buffered draw cmds buffer
    // Pipeline<1> prepareViews;

    Impl(const Config &cfg, vk::Device &dev, vk::MemoryAllocator &mem, 
         VkPipelineCache cache, VkDescriptorSet asset_set);
};

BatchRendererProto::Impl::Impl(const Config &cfg,
                               vk::Device &dev,
                               vk::MemoryAllocator &mem,
                               VkPipelineCache pipeline_cache, 
                               VkDescriptorSet asset_set)
    : dev(dev), mem(mem), pipelineCache(pipeline_cache),
      maxNumViews(cfg.numWorlds * cfg.maxViewsPerWorld),
      targets(makeLayeredTargets(cfg.renderWidth, cfg.renderHeight,
                                 maxNumViews, dev, mem)),
      prepareViews(makeComputePipeline(dev, pipelineCache, sizeof(shader::PrepareViewPushConstant), 2+consts::numDrawCmdBuffers, "prepare_views.hlsl")),
      viewBatches(consts::numDrawCmdBuffers),
      batchFrames(2),
      assetSet(asset_set)
{
    for (uint32_t i = 0; i < consts::numDrawCmdBuffers; i++) {
        makeViewBatch(dev, &viewBatches[i], mem, prepareViews.descPool.makeSet());
    }

    for (uint32_t i = 0; i < 2; i++) {
        makeBatchFrame(dev, &batchFrames[i], mem, cfg.numWorlds, 
            cfg.maxInstancesPerWorld, cfg.maxViewsPerWorld, prepareViews.descPool.makeSet());
    }
}

BatchRendererProto::BatchRendererProto(const Config &cfg,
                                       vk::Device &dev,
                                       vk::MemoryAllocator &mem,
                                       VkPipelineCache pipeline_cache,
                                      VkDescriptorSet asset_set)
    : impl(std::make_unique<Impl>(cfg, dev, mem, pipeline_cache,asset_set))
{
}

BatchRendererProto::~BatchRendererProto()
{
}


void issuePrepareViewsPipeline(vk::Device& dev, VkCommandBuffer& draw_cmd, Pipeline<1>& prepare_views, BatchFrame& frame, ViewBatch& batch,
    VkDescriptorSet& assetSetPrepareView, uint32_t num_worlds, uint32_t num_views, uint32_t view_start) {

    dev.dt.cmdFillBuffer(draw_cmd, batch.drawBuffer.buffer, 0,
        sizeof(shader::DrawCmd) * consts::maxDrawsPerLayeredImage,
        0);

    VkMemoryBarrier copy_barrier{
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

    dev.dt.cmdBindPipeline(draw_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        prepare_views.hdls[0]);

    std::array view_gen_descriptors {
        frame.viewInstanceSet,
        batch.drawBufferSet,
        assetSetPrepareView,
    };

    dev.dt.cmdBindDescriptorSets(draw_cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        prepare_views.layout, 0,
        view_gen_descriptors.size(),
        view_gen_descriptors.data(),
        0, nullptr);

    shader::PrepareViewPushConstant view_push_const{
        num_views, view_start, num_worlds
    };

    dev.dt.cmdPushConstants(draw_cmd, prepare_views.layout,
        VK_SHADER_STAGE_COMPUTE_BIT, 0,
        sizeof(shader::PrepareViewPushConstant), &view_push_const);

    uint32_t num_workgroups = num_views;
    dev.dt.cmdDispatch(draw_cmd, num_workgroups, 1, 1);

    VkMemoryBarrier cull_draw_barrier{
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

void issueDraws(vk::Device& dev, Pipeline<1>& object_draw_, BatchImportedBuffers& buffers, VkCommandBuffer& draw_cmd, vk::LocalBuffer& command_buffer) {
    /*dev.dt.cmdBindPipeline(draw_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
        object_draw_.hdls[0]);

    std::array draw_descriptors {
        frame.drawShaderSet,
        asset_set_draw_,
        asset_set_mat_tex_
    };

    dev.dt.cmdBindDescriptorSets(draw_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
        object_draw_.layout, 0,
        draw_descriptors.size(),
        draw_descriptors.data(),
        0, nullptr);

    DrawPushConst draw_const{
        (uint32_t)cfg.viewIDX,
    };

    dev.dt.cmdPushConstants(draw_cmd, object_draw_.layout,
        VK_SHADER_STAGE_VERTEX_BIT |
        VK_SHADER_STAGE_FRAGMENT_BIT, 0,
        sizeof(DrawPushConst), &draw_const);

    dev.dt.cmdBindIndexBuffer(draw_cmd, loaded_assets_[0].buf.buffer,
        loaded_assets_[0].idxBufferOffset,
        VK_INDEX_TYPE_UINT32);

    VkViewport viewport{
        0,
        0,
        (float)fb_width_,
        (float)fb_height_,
        0.f,
        1.f,
    };

    dev.dt.cmdSetViewport(draw_cmd, 0, 1, &viewport);

    VkRect2D scissor{
        { 0, 0 },
        { fb_width_, fb_height_ },
    };

    dev.dt.cmdSetScissor(draw_cmd, 0, 1, &scissor);


    VkRenderPassBeginInfo render_pass_info;
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass_info.pNext = nullptr;
    render_pass_info.renderPass = render_pass_;
    render_pass_info.framebuffer = frame.fb.hdl;
    render_pass_info.clearValueCount = fb_clear_.size();
    render_pass_info.pClearValues = fb_clear_.data();
    render_pass_info.renderArea.offset = {
        0, 0,
    };
    render_pass_info.renderArea.extent = {
        fb_width_, fb_height_,
    };

    dev.dt.cmdBeginRenderPass(draw_cmd, &render_pass_info,
        VK_SUBPASS_CONTENTS_INLINE);

    dev.dt.cmdDrawIndexedIndirect(draw_cmd,
        frame.renderInput.buffer,
        frame.drawCmdOffset,
        num_instances * 10,
        sizeof(DrawCmd));

    dev.dt.cmdEndRenderPass(draw_cmd);

    { // Issue deferred lighting pass - separate function - this is becoming crazy
        issueLightingPass(dev, frame, deferred_lighting_, draw_cmd, cam, cfg.viewIDX);
    }*/
}

void BatchRendererProto::renderViews(VkCommandBuffer& draw_cmd, BatchRenderInfo info) {
    uint32_t batch_index = 0;
    uint32_t frame_index = 0;

    uint32_t num_views = info.numViews;
    int offset = 0;

    while (num_views > 0) {
        int batch_size = std::min(impl->dev.maxNumLayersPerImage, num_views);
        issuePrepareViewsPipeline(impl->dev, draw_cmd, impl->prepareViews, impl->batchFrames[frame_index],
            impl->viewBatches[batch_index], impl->assetSet, info.numWorlds,
            batch_size, offset);

        //Finish rest of draws for the frame

        offset += batch_size;
        num_views -= batch_size;
        batch_index = ((batch_index + 1) % impl->viewBatches.size());
    }

}
BatchImportedBuffers &BatchRendererProto::getImportedBuffers(uint32_t frame_id) {
    return impl->batchFrames[frame_id].buffers;
}

}
