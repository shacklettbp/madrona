#include <madrona/taskgraph.hpp>
#include <madrona/mw_gpu/megakernel_consts.hpp>

namespace madrona {

TaskGraph::Builder::Builder(int32_t max_num_nodes,
                            int32_t max_node_datas,
                            int32_t max_num_dependencies)
    : staged_((StagedNode *)rawAlloc(sizeof(StagedNode) * max_num_nodes)),
      num_nodes_(0),
      node_datas_((NodeData *)rawAlloc(sizeof(NodeData) * max_node_datas)),
      num_datas_(0),
      all_dependencies_((NodeID *)rawAlloc(sizeof(NodeID) * max_num_dependencies)),
      num_dependencies_(0)
{}

TaskGraph::Builder::~Builder()
{
    rawDealloc(all_dependencies_);
    rawDealloc(node_datas_),
    rawDealloc(staged_);
}

TaskGraph::NodeID TaskGraph::Builder::registerNode(
    uint32_t data_idx,
    uint32_t fixed_count,
    uint32_t num_threads_per_invocation,
    uint32_t func_id,
    Span<const TaskGraph::NodeID> dependencies,
    Optional<NodeID> parent_node)
{
    assert(consts::numMegakernelThreads % num_threads_per_invocation == 0);

    uint32_t offset = num_dependencies_;
    uint32_t num_deps = dependencies.size();

    num_dependencies_ += num_deps;

    for (int i = 0; i < (int)num_deps; i++) {
        all_dependencies_[offset + i] = dependencies[i];
    }

    int32_t node_idx = num_nodes_++;

    new (&staged_[node_idx]) StagedNode {
        {
            data_idx,
            fixed_count,
            func_id,
            0,
            num_threads_per_invocation,
            0,
            0,
            0,
        },
        parent_node.has_value() ? parent_node->id : -1,
        offset,
        num_deps,
    };

    return NodeID {
        node_idx,
    };
}

void TaskGraph::Builder::build(TaskGraph *out)
{
    assert(staged_[0].numDependencies == 0);
    Node *sorted_nodes = 
        (Node *)rawAlloc(sizeof(Node) * num_nodes_);
    bool *queued = (bool *)rawAlloc(num_nodes_ * sizeof(bool));
    int32_t *num_children = (int32_t *)rawAlloc(num_nodes_ * sizeof(uint32_t));

    uint32_t sorted_idx = 0;
    auto enqueueInSorted = [&](const Node &node) {
        new (&sorted_nodes[sorted_idx++]) Node {
            node.dataIDX,
            node.fixedCount,
            node.funcID,
            node.numChildren,
            node.numThreadsPerInvocation,
            0, 0, 0,
        };
    };

    enqueueInSorted(staged_[0].node);

    queued[0] = true;
    num_children[0] = 0;

    uint32_t num_remaining_nodes = num_nodes_ - 1;

    for (int32_t i = 1; i < (int32_t)num_nodes_; i++) {
        queued[i] = false;
        num_children[i] = 0;
    }

    for (int32_t i = 0; i < (int32_t)num_nodes_; i++) {
        auto &staged = staged_[i];

        int32_t parent_id = staged.parentID;
        if (parent_id != -1) {
            num_children[parent_id] += 1;
        }
    }

    while (num_remaining_nodes > 0) {
        uint32_t cur_node_idx;
        for (cur_node_idx = 0; queued[cur_node_idx]; cur_node_idx++) {}

        StagedNode &cur_staged = staged_[cur_node_idx];

        bool dependencies_satisfied = true;
        for (uint32_t dep_offset = 0; dep_offset < cur_staged.numDependencies;
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

    rawDealloc(num_children);
    rawDealloc(queued);

    auto tg_datas = (NodeData *)rawAlloc(sizeof(NodeData) * num_datas_);
    memcpy(tg_datas, node_datas_, sizeof(NodeData) * num_datas_);

    new (out) TaskGraph(sorted_nodes, num_nodes_, tg_datas);
}

ClearTmpNodeBase::ClearTmpNodeBase(uint32_t archetype_id)
    : NodeBase(),
      archetypeID(archetype_id)
{}

void ClearTmpNodeBase::run(int32_t)
{
    StateManager *state_mgr = mwGPU::getStateManager();
    state_mgr->clearTemporaries(archetypeID);
}

TaskGraph::NodeID ClearTmpNodeBase::addToGraph(
    TaskGraph::Builder &builder,
    Span<const TaskGraph::NodeID> dependencies,
    uint32_t archetype_id)
{
    return builder.addOneOffNode<ClearTmpNodeBase>(dependencies, archetype_id);
}

RecycleEntitiesNode::RecycleEntitiesNode()
    : NodeBase(),
      recycleBase(0)
{}

void RecycleEntitiesNode::run(int32_t invocation_idx)
{
    mwGPU::getStateManager()->recycleEntities(
        invocation_idx, recycleBase);
}

uint32_t RecycleEntitiesNode::numInvocations()
{
    auto [recycle_base, num_deleted] =
        mwGPU::getStateManager()->fetchRecyclableEntities();

    if (num_deleted > 0) {
        recycleBase = recycle_base;
    }

    return num_deleted;
}

TaskGraph::NodeID RecycleEntitiesNode::addToGraph(
    TaskGraph::Builder &builder,
    Span<const TaskGraph::NodeID> dependencies)
{
    return builder.addDynamicCountNode<RecycleEntitiesNode>(dependencies, 1);
}

void ResetTmpAllocNode::run(int32_t)
{
    mwGPU::TmpAllocator::get().reset();
}

TaskGraph::NodeID ResetTmpAllocNode::addToGraph(
    TaskGraph::Builder &builder,
    Span<const TaskGraph::NodeID> dependencies)
{
    return builder.addOneOffNode<ResetTmpAllocNode>(dependencies);
}

CompactArchetypeNodeBase::CompactArchetypeNodeBase(uint32_t archetype_id)
    : NodeBase(),
      archetypeID(archetype_id)
{}

void CompactArchetypeNodeBase::run(int32_t invocation_idx)
{
#if 0
    uint32_t archetype_id = data.compactArchetype.archetypeID;
    StateManager *state_mgr = mwGPU::getStateManager();
#endif

    // Actually compact
    assert(false);
}

uint32_t CompactArchetypeNodeBase::numInvocations()
{
    assert(false);
    return mwGPU::getStateManager()->numArchetypeRows(archetypeID);
}

TaskGraph::NodeID CompactArchetypeNodeBase::addToGraph(
    TaskGraph::Builder &builder,
    Span<const TaskGraph::NodeID> dependencies,
    uint32_t archetype_id)
{
    return builder.addDynamicCountNode<CompactArchetypeNodeBase>(
        dependencies, 1, archetype_id);
}

}
