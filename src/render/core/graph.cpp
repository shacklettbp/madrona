#include <madrona/render/graph.hpp>

namespace madrona::render {

struct RenderGraph::Task {
    void (*fn)(void *, GPU &);
    void *data;
};

RenderGraphBuilder::RenderGraphBuilder(StackAlloc &alloc)
    : alloc_(&alloc),
      alloc_start_(alloc_->push()),
      total_resources_(0),
      task_list_head_(&fake_task_),
      task_list_tail_(&fake_task_),
      rsrc_list_head_(&fake_rsrc_),
      rsrc_list_tail_(&fake_rsrc_),
      fake_task_ {}
      fake_rsrc_ {}
{
}

// Goals: Read through the task graph, perform barrier batching and
// resource aliasing. For now it just puts a full barrier whenever necessary
RenderGraph RenderGraphBuilder::build(GPU &gpu)
{
    TaskDesc *cur_task = task_list_head_->next;
    assert(cur != nullptr);

    CountT num_final_tasks = 0;
    CountT total_data_bytes = 0;
    while (cur_task != nullptr) {
        num_final_tasks += 1;
        total_data_bytes += cur->numDataBytes;

        // First, check if this task can be run without a
        // (are any of the read resources "live")

        bool needs_barrier = false;
        for (TaskResource rsrc_hdl : cur_task->readResources) {
            LogicalResource *rsrc = rsrc_hdl.rsrc;

            if (rsrc->isLive) {
                needs_barrier = true;
                break;
            }
        }

        if (needs_barrier) {
            LogicalResource *cur_rsrc = rsrc_list_head_->next;

            while (cur_rsrc != nullptr) {
                cur_rsrc->isLive = false;
                cur_rsrc = cur_rsrc->next;
            }
        }

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
