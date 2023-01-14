#include <madrona/taskgraph.hpp>
#include <madrona/crash.hpp>

#include <sched.h>
#include <unistd.h>

#include "worker_init.hpp"

namespace madrona {

TaskGraph::Builder::Builder(Context &ctx)
    : ctx_(&ctx),
      staged_(0),
      node_datas_(0),
      all_dependencies_(0)
{}

TaskGraph::NodeID TaskGraph::Builder::registerNode(
    uint32_t data_idx,
    void (*fn)(NodeBase *, Context *),
    Span<const NodeID> dependencies,
    Optional<NodeID> parent_node)
{
    CountT dependency_offset = all_dependencies_.size();

    for (NodeID node_id : dependencies) {
        all_dependencies_.push_back(node_id);
    }

    staged_.push_back({
        .node = {
            .fn = fn,
            .dataIDX = data_idx,
            .numChildren = 0,
        },
        .parentID = parent_node.has_value() ? int32_t(parent_node->id) : -1,
        .dependencyOffset = uint32_t(dependency_offset),
        .numDependencies = uint32_t(dependencies.size()),
    });

    return NodeID {
        uint32_t(staged_.size() - 1),
    };
}

TaskGraph TaskGraph::Builder::build()
{
    assert(staged_[0].numDependencies == 0);

    HeapArray<Node> sorted_nodes(staged_.size());
    HeapArray<bool> queued(staged_.size());
    HeapArray<int32_t> num_children(staged_.size());

    int32_t sorted_idx = 0;
    auto enqueueInSorted = [&](const Node &node) {
        new (&sorted_nodes[sorted_idx++]) Node(node);
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

    HeapArray<NodeData> data_cpy(node_datas_.size());
    memcpy(data_cpy.data(), node_datas_.data(),
           node_datas_.size() * sizeof(NodeData));

    return TaskGraph(std::move(sorted_nodes), std::move(data_cpy));
}

TaskGraph::TaskGraph(HeapArray<Node> &&sorted_nodes,
                     HeapArray<NodeData> &&node_datas)
    : sorted_nodes_(std::move(sorted_nodes)),
      node_datas_(std::move(node_datas))
{}

void TaskGraph::run(Context *ctx)
{
    for (const Node &node : sorted_nodes_) {
        node.fn((NodeBase *)(&node_datas_[node.dataIDX].userData[0]), ctx);
    }
}

TaskGraph::NodeID ResetTmpAllocNode::addToGraph(
    Context &,
    TaskGraph::Builder &builder,
    Span<const TaskGraph::NodeID> dependencies)
{
    return builder.addDefaultNode<ResetTmpAllocNode>(dependencies);
}

}
