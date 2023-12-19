#include <madrona/render/graph.hpp>
#include <cassert>

namespace madrona::render {

struct RenderGraph::Task {
    void (*fn)(void *, GPU &, CommandBuffer);
    void *data;
};

RenderGraphBuilder::RenderGraphBuilder(StackAlloc &alloc)
    : alloc_(&alloc),
      alloc_start_(alloc_->push()),
      task_list_head_(alloc_->alloc<TaskDesc>()),
      task_list_tail_(task_list_head_),
      rsrc_list_head_(alloc_->alloc<LogicalResource>()),
      rsrc_list_tail_(rsrc_list_head_)
{
    task_list_head_->next = nullptr;
    rsrc_list_head_->next = nullptr;
}

bool RenderGraphBuilder::checkHazards(TaskDesc *task)
{
    using Type = TaskDesc::Type;

    bool hazard = false;
    switch (task->type) {
    case Type::Raster: {
        hazard |= checkHazards(task->raster.vert.read);
        hazard |= checkHazards(task->raster.frag.read);
        hazard |= checkHazards(task->raster.attachments.readwrite);
    } break;
    case Type::Compute: {
        hazard |= checkHazards(task->compute.read);
        hazard |= checkHazards(task->compute.readwrite);
    } break;
    case Type::Copy: {
        hazard |= checkHazards(task->copy.read);
    } break;
    default: break;
    }

    return hazard;
}

bool RenderGraphBuilder::checkHazards(Span<const TaskResource> resources)
{
    bool hazard = false;
    for (TaskResource rsrc : resources) {
        hazard |= getResource(rsrc)->writeHazard;
    }

    return hazard;
}

void RenderGraphBuilder::markHazards(TaskDesc *task)
{
    using Type = TaskDesc::Type;
    switch (task->type) {
    case Type::Raster: {
        markHazards(task->raster.attachments.readwrite);
        markHazards(task->raster.attachments.write);
        markHazards(task->raster.attachments.clear);
    } break;
    case Type::Compute: {
        markHazards(task->compute.write);
        markHazards(task->compute.readwrite);
    } break;
    case Type::Copy: {
        markHazards(task->copy.write);
    } break;
    default: break;
    }
}

void RenderGraphBuilder::markHazards(Span<const TaskResource> resources)
{
    for (TaskResource rsrc : resources) {
        getResource(rsrc)->writeHazard = true;
    }
}

TaskDesc * RenderGraphBuilder::addBarrier()
{
    TaskDesc *barrier_task = addTaskCommon([](
            GPU &gpu, CommandBuffer cmd) {
        gpu.debugFullBarrier(cmd);
    });
    barrier_task->type = TaskDesc::Type::Barrier;

    return barrier_task;
}

// Goals: Read through the task graph, perform barrier batching and
// resource aliasing. For now it just puts a full barrier whenever necessary
RenderGraph RenderGraphBuilder::build(GPU &gpu, CountT num_inflight)
{
    // Head is a fake entry
    TaskDesc *prev_task = task_list_head_;

    TaskDesc *cur_task = prev_task->next;
    assert(cur_task != nullptr);

    CountT total_num_tasks = 0;
    CountT total_param_blocks = 0;
    CountT total_data_bytes = 0;
    while (cur_task != nullptr) {
        total_num_tasks += 1;
        total_data_bytes =
            utils::roundUpPow2(total_data_bytes, cur_task->dataAlignment);
        total_data_bytes += cur_task->numDataBytes;

        // First, check if this task can be run without a barrier
        // (are any of the read resources "live")

        bool needs_barrier = checkHazards(cur_task);
        if (needs_barrier) {
            TaskDesc *barrier_task = addBarrier(prev_task, cur_task);
            prev_task->next = barrier_task;
            barrier_task->next = cur_task;

            total_num_tasks += 1;
            total_data_bytes = utils::roundUpPow2(
                total_data_bytes, barrier_task->dataAlignment);
            total_data_bytes += barrier_task->numDataBytes;
        }

        markHazards(cur_task);
        prev_task = cur_task;
        cur_task = cur_task->next;
    }

    CountT final_num_textures = 0;
    CountT final_num_buffers = 0;
    LogicalResource *cur_rsrc = rsrc_list_head_->next;
    while (cur_rsrc != nullptr) {
        switch (cur_rsrc->type) {
        case LogicalResource::Type::Texture2D: {
            auto &tex2D = cur_rsrc->tex2D;

            tex2D.allocatedHandle = gpu.makeTex2D(
                tex2D.desc.width, tex2D.desc.height, tex2D.desc.fmt);

            final_num_textures += 1;
        } break;
        case LogicalResource::Type::Buffer: {
            auto &buf = cur_rsrc->buffer;

            buf.allocatedHandle = gpu.makeBuffer(buf.desc.numBytes);

            final_num_buffers += 1;
        } break;
        }

        cur_rsrc = cur_rsrc->next;
    }

    int64_t graph_buffer_offsets[3];
    CountT total_graph_bytes = utils::computeBufferOffsets({
        total_data_bytes,
        total_num_tasks * sizeof(RenderGraph::Task),
        final_num_textures * sizeof(TextureHandle),
        final_num_buffers * sizeof(BufferHandle),
    }, graph_buffer_offsets, 8);

    auto *graph_data_buffer = (char *)malloc(total_graph_bytes);

    auto *tasks_out =
        (RenderGraph::Task *)(graph_data_buffer + graph_buffer_offsets[0]);
    auto *textures_out =
        (TextureHandle *)(graph_data_buffer + graph_buffer_offsets[1]);
    auto *buffers_out = 
        (BufferHandle *)(graph_data_buffer + graph_buffer_offsets[2]);

    cur_task = task_list_head_->next;
    CountT cur_task_idx = 0;
    uint64_t cur_data_offset = 0;
    while (cur_task != nullptr) {
        cur_data_offset = utils::roundUpPow2(cur_data_offset,
                                             cur_task->dataAlignment);
        char *data_dst = graph_data_buffer + cur_data_offset;
        memcpy(data_dst, cur_task->data, cur_task->numDataBytes);

        cur_data_offset += cur_task->numDataBytes;

        tasks_out[cur_task_idx++] = {
            cur_task->fn,
            data_dst,
        };
    }

    cur_rsrc = rsrc_list_head_->next;
    CountT cur_texture_idx = 0;
    CountT cur_buffer_idx = 0;
    while (cur_rsrc != nullptr) {
        switch (cur_rsrc->type) {
        case LogicalResource::Type::Texture2D: {
            textures_out[cur_texture_idx++] =
                cur_rsrc->tex2D.allocatedHandle;
        } break;
        case LogicalResource::Type::Buffer: {
            buffers_out[cur_buffer_idx++] =
                cur_rsrc->buffer.allocatedHandle;
        } break;
        }

        cur_rsrc = cur_rsrc->next;
    }

    RenderGraph render_graph;

    // Reset the builder
    task_list_head_->next = nullptr;
    num_resources_ = 0;
    alloc_->pop(alloc_start_);

    return render_graph;
}

RenderGraph::RenderGraph(void *data_buffer,
                         Span<const Task> tasks,
                         Span<const TextureHandle> textures,
                         Span<const BufferHandle> buffers)
    : data_buffer_(data_buffer),
      tasks_(tasks),
      textures_(textures),
      buffers_(buffers)
{}

RenderGraph::~RenderGraph()
{
    free(data_buffer_);
}

}
