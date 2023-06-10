#include <madrona/render/graph.hpp>

namespace madrona::render {

RenderGraphBuilder::RenderGraphBuilder(StackAlloc &alloc)
    : alloc_(&alloc),
      alloc_start_(alloc_->push()),
      task_list_head_(&empty_),
      task_list_tail_(&empty_)
{
    empty_.next = nullptr;
}

RenderGraph RenderGraphBuilder::build(GPU &gpu)
{
    alloc_->pop(alloc_start_);
    return RenderGraph();
}

}
