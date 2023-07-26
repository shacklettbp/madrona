#include <madrona/taskgraph.hpp>
#include <madrona/crash.hpp>
#include <madrona/macros.hpp>
#include <madrona/taskgraph_builder.hpp>

#include "worker_init.hpp"

namespace madrona {

TaskGraphBuilder::TaskGraphBuilder(const WorkerInit &init)
    : state_mgr_(init.stateMgr),
      state_cache_(init.stateCache),
#ifdef MADRONA_MW_MODE
      world_id_(init.worldID),
#endif
      staged_(0),
      node_datas_(0),
      all_dependencies_(0)
{}

TaskGraphNodeID TaskGraphBuilder::registerNode(
    uint32_t data_idx,
    void (*fn)(NodeBase *, Context *, TaskGraph *),
    Span<const TaskGraphNodeID> dependencies,
    Optional<TaskGraphNodeID> parent_node)
{
    CountT dependency_offset = all_dependencies_.size();

    for (TaskGraphNodeID node_id : dependencies) {
        all_dependencies_.push_back(node_id);
    }

    staged_.push_back(StagedNode {
        .node = {
            .fn = fn,
            .dataIDX = data_idx,
            .numChildren = 0,
        },
        .parentID = parent_node.has_value() ? int32_t(parent_node->id) : -1,
        .dependencyOffset = uint32_t(dependency_offset),
        .numDependencies = uint32_t(dependencies.size()),
    });

    return TaskGraphNodeID {
        uint32_t(staged_.size() - 1),
    };
}

TaskGraph TaskGraphBuilder::build()
{
    assert(staged_[0].numDependencies == 0);

    HeapArray<TaskGraph::Node> sorted_nodes(staged_.size());
    HeapArray<bool> queued(staged_.size());
    HeapArray<int32_t> num_children(staged_.size());

    int32_t sorted_idx = 0;
    auto enqueueInSorted = [&](const TaskGraph::Node &node) {
        new (&sorted_nodes[sorted_idx++]) TaskGraph::Node(node);
    };

    enqueueInSorted(staged_[0].node);

    queued[0] = true;

    CountT num_remaining_nodes = staged_.size() - 1;

    for (CountT i = 1; i < staged_.size(); i++) {
        queued[i] = false;
        num_children[i] = 0;
    }

    for (CountT i = 0; i < staged_.size(); i++) {
        auto &staged = staged_[i];

        int32_t parent_id = staged.parentID;
        if (parent_id != -1) {
            num_children[parent_id] += 1;
        }
    }

    while (num_remaining_nodes > 0) {
        CountT cur_node_idx;
        for (cur_node_idx = 0; queued[cur_node_idx]; cur_node_idx++) {}

        StagedNode &cur_staged = staged_[cur_node_idx];

        bool dependencies_satisfied = true;
        for (CountT dep_offset = 0;
             dep_offset < (CountT)cur_staged.numDependencies;
             dep_offset++) {
            uint32_t dep_node_idx =
                all_dependencies_[cur_staged.dependencyOffset + dep_offset].id;
            if (!queued[dep_node_idx]) {
                dependencies_satisfied = false;
                break;
            }
        }

        if (dependencies_satisfied) {
            queued[cur_node_idx] = true;
            enqueueInSorted(cur_staged.node);
            num_remaining_nodes--;
        }
    }

    HeapArray<TaskGraph::NodeData> data_cpy(node_datas_.size());
    memcpy(data_cpy.data(), node_datas_.data(),
           node_datas_.size() * sizeof(TaskGraph::NodeData));

    return TaskGraph(state_mgr_, state_cache_, MADRONA_MW_COND(world_id_,)
        std::move(sorted_nodes), std::move(data_cpy));
}

TaskGraph::TaskGraph(StateManager *state_mgr,
                     StateCache *state_cache,
                     MADRONA_MW_COND(uint32_t world_id,) 
                     HeapArray<Node> &&sorted_nodes,
                     HeapArray<NodeData> &&node_datas)
    : state_mgr_(state_mgr),
      state_cache_(state_cache),
#ifdef MADRONA_MW_MODE
      cur_world_id_(world_id),
#endif
      sorted_nodes_(std::move(sorted_nodes)),
      node_datas_(std::move(node_datas))
{}

void TaskGraph::run(Context *ctx)
{
    for (const Node &node : sorted_nodes_) {
        node.fn((NodeBase *)(&node_datas_[node.dataIDX].userData[0]),
                ctx, this);
    }
}

void TaskGraph::resetTmpAlloc()
{
    state_mgr_->resetTmpAlloc(MADRONA_MW_COND(cur_world_id_));
}

TaskGraphNodeID ResetTmpAllocNode::addToGraph(
    StateManager &,
    TaskGraphBuilder &builder,
    Span<const TaskGraphNodeID> dependencies)
{
    return builder.addDefaultNode<ResetTmpAllocNode>(dependencies);
}

}
