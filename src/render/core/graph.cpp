#include <madrona/render/graph.hpp>
#include <cassert>

namespace madrona::render {

struct RenderGraph::Task {
    void (*fn)(void *, GPU &);
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

void RenderGraphBuilder::addBarrier(TaskDesc *prev, TaskDesc *next)
{
    TaskDesc *barrier_task = addTaskCommon([](
            GPU &gpu, CommandBuffer cmd) {
        gpu.debugFullBarrier(cmd);
    });
    
    barrier_task->type = TaskDesc::Type::Barrier;

    prev->next = barrier_task;
    barrier_task->next = next;
}

// Goals: Read through the task graph, perform barrier batching and
// resource aliasing. For now it just puts a full barrier whenever necessary
RenderGraph RenderGraphBuilder::build(GPU &gpu)
{
    // Head is a fake entry
    TaskDesc *prev_task = task_list_head_;

    TaskDesc *cur_task = prev_task->next;
    assert(cur_task != nullptr);

    CountT num_final_tasks = 0;
    CountT total_data_bytes = 0;
    while (cur_task != nullptr) {
        num_final_tasks += 1;
        total_data_bytes += cur_task->numDataBytes;

        // First, check if this task can be run without a barrier
        // (are any of the read resources "live")

        bool needs_barrier = checkHazards(cur_task);
        if (needs_barrier) {
            addBarrier(prev_task, cur_task);
        }

        markHazards(cur_task);
        prev_task = cur_task;
        cur_task = cur_task->next;
    }

    RenderGraph render_graph;

    // Reset the builder
    task_list_head_->next = nullptr;
    num_resources_ = 0;
    alloc_->pop(alloc_start_);

    return render_graph;
}

RenderGraph::RenderGraph(void *data_buffer, Span<const Task> tasks)
    : data_buffer_(data_buffer),
      tasks_(tasks)
{}

RenderGraph::~RenderGraph()
{
    free(data_buffer_);
}

}
