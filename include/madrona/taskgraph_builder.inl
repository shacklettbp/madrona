#pragma once

namespace madrona {

template <typename NodeT>
TaskGraphNodeID TaskGraphBuilder::addToGraph(
    Span<const TaskGraphNodeID> dependencies)
{
    return NodeT::addToGraph(*state_mgr_, *this, dependencies);
}

template <typename NodeT, typename... Args> 
TaskGraphBuilder::TypedDataID<NodeT> TaskGraphBuilder::constructNodeData(
    Args && ...args)
{
    static_assert(sizeof(NodeT) <= TaskGraph::maxNodeDataBytes);
    static_assert(alignof(NodeT) <= TaskGraph::maxNodeDataAlign);

    CountT data_idx = node_datas_.uninit_back();
    new (&node_datas_[data_idx]) NodeT(std::forward<Args>(args)...);

    return TypedDataID<NodeT> {
        DataID { int32_t(data_idx) },
    };
}

template <auto fn, typename NodeT>
TaskGraphNodeID TaskGraphBuilder::addNodeFn(
        TypedDataID<NodeT> data,
        Span<const TaskGraphNodeID> dependencies,
        Optional<TaskGraphNodeID> parent_node)
{
    return registerNode(uint32_t(data.id), [](NodeBase *node_data,
                                              Context *ctx,
                                              TaskGraph *task_graph) {
            std::invoke(fn, ((NodeT *)node_data), *ctx, *task_graph);
        },
        dependencies,
        parent_node);
}

template <typename NodeT, typename... Args>
TaskGraphNodeID TaskGraphBuilder::addDefaultNode(
    Span<const TaskGraphNodeID> dependencies,
    Args && ...args)
{
    auto data_id = constructNodeData<NodeT>(
        std::forward<Args>(args)...);
    return addNodeFn<&NodeT::run>(data_id, dependencies,
                                  Optional<TaskGraphNodeID>::none());
}

template <typename NodeT>
NodeT & TaskGraphBuilder::getDataRef(TypedDataID<NodeT> data_id)
{
    return *(NodeT *)node_datas_[data_id.id].userData;
}

template <EnumType EnumT>
TaskGraphBuilder & TaskGraphManager::init(EnumT taskgraph_id)
{
    return init(static_cast<uint32_t>(taskgraph_id));
}

template <typename ContextT, auto Fn, typename ...ComponentTs>
ParallelForNode<ContextT, Fn, ComponentTs...>::ParallelForNode(
        Query<ComponentTs...> &&query)
    : query_(std::move(query))
{}

template <typename ContextT, auto Fn, typename ...ComponentTs>
void ParallelForNode<ContextT, Fn, ComponentTs...>::run(
    Context &ctx_base, TaskGraph &taskgraph)
{
    ContextT &ctx = static_cast<ContextT &>(ctx_base);
    taskgraph.iterateQuery(ctx, query_, Fn); 
}

template <typename ContextT, auto Fn, typename ...ComponentTs>
TaskGraphNodeID
ParallelForNode<ContextT, Fn, ComponentTs...>::addToGraph(
    StateManager &state_mgr,
    TaskGraphBuilder &builder,
    Span<const TaskGraphNodeID> dependencies)
{
    using NodeT = ParallelForNode<ContextT, Fn, ComponentTs...>;

    auto query = state_mgr.query<ComponentTs...>();
    return builder.addDefaultNode<NodeT>(dependencies, std::move(query));
}

void ResetTmpAllocNode::run(Context &, TaskGraph &taskgraph)
{
    taskgraph.resetTmpAlloc();
}

template <typename ArchetypeT>
void ClearTmpNode<ArchetypeT>::run(Context &, TaskGraph &taskgraph)
{
    taskgraph.clearTemporaries<ArchetypeT>();
}

template <typename ArchetypeT>
TaskGraphNodeID ClearTmpNode<ArchetypeT>::addToGraph(
    StateManager &,
    TaskGraphBuilder &builder,
    Span<const TaskGraphNodeID> dependencies)
{
    return builder.addDefaultNode<ClearTmpNode>(dependencies);
}

template <typename ArchetypeT, typename ComponentT>
TaskGraphNodeID SortArchetypeNode<ArchetypeT, ComponentT>::addToGraph(
        StateManager &state_mgr,
        TaskGraphBuilder &builder,
        Span<const TaskGraphNodeID> dependencies)
{
    return SortArchetypeNodeBase::addToGraph(state_mgr, builder, dependencies,
        TypeTracker::typeID<ArchetypeT>(),
        TypeTracker::typeID<ComponentT>());
}

template <typename ArchetypeT>
TaskGraphNodeID CompactArchetypeNode<ArchetypeT>::addToGraph(
        StateManager &state_mgr,
        TaskGraphBuilder &builder,
        Span<const TaskGraphNodeID> dependencies)
{
    return CompactArchetypeNodeBase::addToGraph(state_mgr, builder, dependencies,
        TypeTracker::typeID<ArchetypeT>());
}

}
