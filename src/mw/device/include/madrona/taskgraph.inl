#pragma once

namespace madrona {

namespace mwGPU {

TaskGraph & getTaskGraph(uint32_t taskgraph_idx)
{
    return ((TaskGraph *)mwGPU::GPUImplConsts::get().taskGraph)[taskgraph_idx];
}

template <typename NodeT, auto fn>
__attribute__((used, always_inline))
inline void userEntry(NodeBase *data_ptr, int32_t invocation_idx)
{
    auto node = (NodeT *)data_ptr;

    std::invoke(fn, node, invocation_idx);
}

template <typename NodeT, auto fn>
struct UserFuncIDBase {
    static uint32_t id;
};

template <typename NodeT,
          auto fn,
          decltype(userEntry<NodeT, fn>) = userEntry<NodeT, fn>>
struct UserFuncID : UserFuncIDBase<NodeT, fn> {};

}

template <typename ContextT, bool = false>
struct TaskGraph::WorldTypeExtract {
    using type = typename ContextT::WorldDataT;
};

template <bool ignore>
struct TaskGraph::WorldTypeExtract<Context, ignore> {
    using type = WorldBase;
};

template <typename NodeT, typename... Args> 
TaskGraph::TypedDataID<NodeT> TaskGraph::Builder::constructNodeData(
    Args && ...args)
{
    static_assert(sizeof(NodeT) <= maxNodeDataBytes);
    static_assert(alignof(NodeT) <= maxNodeDataBytes);

    int32_t data_idx = num_datas_++;
    assert(num_datas_ <= max_num_node_datas_);
    new (&node_datas_[data_idx]) NodeT(std::forward<Args>(args)...);

    return TypedDataID<NodeT> {
        DataID { data_idx },
    };
}

template <auto fn, typename NodeT>
TaskGraph::NodeID TaskGraph::Builder::addNodeFn(
        TypedDataID<NodeT> data,
        Span<const NodeID> dependencies,
        Optional<NodeID> parent_node,
        uint32_t fixed_num_invocations,
        uint32_t num_threads_per_invocation)
{
    using namespace mwGPU;

    uint32_t func_id = mwGPU::UserFuncID<NodeT, fn>::id;

    return registerNode(uint32_t(data.id),
                        fixed_num_invocations,
                        num_threads_per_invocation,
                        func_id,
                        dependencies,
                        parent_node);
}

template <typename NodeT, int32_t count, typename... Args>
TaskGraph::NodeID TaskGraph::Builder::addOneOffNode(
    Span<const NodeID> dependencies,
    Args && ...args)
{
    auto data_id = constructNodeData<NodeT>(
        std::forward<Args>(args)...);
    return addNodeFn<&NodeT::run>(data_id, dependencies,
                                  Optional<NodeID>::none(), count);
}

template <typename NodeT>
void TaskGraph::Builder::dynamicCountWrapper(NodeT *node, int32_t)
{
    int32_t num_invocations = node->numInvocations();
    node->numDynamicInvocations = num_invocations;
}

template <typename NodeT, typename... Args>
TaskGraph::NodeID TaskGraph::Builder::addDynamicCountNode(
    Span<const NodeID> dependencies,
    uint32_t num_threads_per_invocation,
    Args && ...args)
{
    auto data_id = constructNodeData<NodeT>(
        std::forward<Args>(args)...);

    NodeID count_node = addNodeFn<&Builder::dynamicCountWrapper<NodeT>>(
        data_id, dependencies, Optional<NodeID>::none(), 1);

    return addNodeFn<&NodeT::run>(data_id, {count_node},
        Optional<NodeID>::none(), 0, num_threads_per_invocation);
}

template <typename NodeT>
TaskGraph::NodeID TaskGraph::Builder::addToGraph(
    Span<const NodeID> dependencies)
{
    return NodeT::addToGraph(*this, dependencies);
}

template <typename NodeT>
NodeT & TaskGraph::Builder::getDataRef(TypedDataID<NodeT> data_id)
{
    return *(NodeT *)node_datas_[data_id.id].userData;
}


inline uint32_t TaskGraph::Builder::getTaskgraphID() const
{
    return taskgraph_id_;
}

template <EnumType EnumT>
TaskGraphBuilder & TaskGraphManager::init(EnumT taskgraph_id)
{
    return init(static_cast<uint32_t>(taskgraph_id));
}

WorldBase * TaskGraph::getWorld(int32_t world_idx)
{
    const auto &consts = mwGPU::GPUImplConsts::get();
    auto world_ptr = (char *)consts.worldDataAddr +
        world_idx * (int32_t)consts.numWorldDataBytes;

    return (WorldBase *)world_ptr;
}

template <typename ContextT>
ContextT TaskGraph::makeContext(WorldID world_id)
{
    using WorldDataT = typename WorldTypeExtract<ContextT>::type;

    auto world = TaskGraph::getWorld(world_id.idx);
    return ContextT((WorldDataT *)world, WorkerInit {
        world_id,
    });
}

template <typename NodeT>
NodeT & TaskGraph::getNodeData(TypedDataID<NodeT> data_id)
{
    return *(NodeT *)node_datas_[data_id.id].userData;
}

template <typename ContextT, auto Fn,
          int32_t threads_per_invocation,
          int32_t items_per_invocation,
          typename ...ComponentTs>
CustomParallelForNode<ContextT, Fn, threads_per_invocation,
                      items_per_invocation, ComponentTs...>::
CustomParallelForNode()
    : NodeBase {},
      query_ref_([]() {
          auto query = mwGPU::getStateManager()->query<ComponentTs...>();
          QueryRef *query_ref = query.getSharedRef();
          query_ref->numReferences.fetch_add_relaxed(1);

          return query_ref;
      }())
{}

template <typename ContextT, auto Fn,
          int32_t threads_per_invocation,
          int32_t items_per_invocation,
          typename ...ComponentTs>
void CustomParallelForNode<ContextT, Fn,
                           threads_per_invocation,
                           items_per_invocation,
                           ComponentTs...>::run(const int32_t invocation_idx)
{
    // Special case the vastly common case
    if constexpr (items_per_invocation == 1) {
        StateManager *state_mgr = mwGPU::getStateManager();

        int32_t cumulative_num_rows = 0;
        state_mgr->iterateArchetypesRaw<sizeof...(ComponentTs)>(query_ref_,
                [&](int32_t num_rows, WorldID *world_column,
                    auto ...raw_ptrs) {
            int32_t tbl_offset = invocation_idx - cumulative_num_rows;
            cumulative_num_rows += num_rows;
            if (tbl_offset >= num_rows) {
                return false;
            }

            WorldID world_id = world_column[tbl_offset];

            // This entity has been deleted but not actually removed from the
            // table yet
            if (world_id.idx == -1) {
                return true;
            }

            ContextT ctx = TaskGraph::makeContext<ContextT>(world_id);

            // The following should work, but doesn't in cuda 11.7 it seems
            // Need to put arguments in tuple for some reason instead
            //Fn(ctx, ((ComponentTs *)raw_ptrs)[tbl_offset] ...);

            cuda::std::tuple typed_ptrs {
                (ComponentTs *)raw_ptrs
                ...
            };

            std::apply([&](auto ...ptrs) {
                Fn(ctx, ptrs[tbl_offset] ...);
            }, typed_ptrs);

            return true;
        });
    } else {
        int32_t base_item_idx = invocation_idx * items_per_invocation;
        int32_t cur_item_offset = 0;

        StateManager *state_mgr = mwGPU::getStateManager();
        int32_t cumulative_num_rows = 0;
        state_mgr->iterateArchetypesRaw<sizeof...(ComponentTs)>(query_ref_,
                [&](int32_t num_rows, WorldID *world_column,
                    auto ...raw_ptrs) {
            int32_t item_idx = base_item_idx + cur_item_offset;
            int32_t tbl_offset = item_idx - cumulative_num_rows;
            cumulative_num_rows += num_rows;

            int32_t launch_size = min(num_rows - tbl_offset,
                                      items_per_invocation);
            if (launch_size <= 0) {
                return false;
            }

            // The following should work, but doesn't in cuda 11.7 it seems
            // Need to put arguments in tuple for some reason instead
            //Fn(ctx, ((ComponentTs *)raw_ptrs)[tbl_offset] ...);

            cuda::std::tuple typed_ptrs {
                (ComponentTs *)raw_ptrs
                ...
            };

            std::apply([&](auto ...ptrs) {
                Fn(world_column + tbl_offset,
                   ptrs + tbl_offset ...,
                   launch_size);
            }, typed_ptrs);

            cur_item_offset += launch_size;
            return cur_item_offset == items_per_invocation;
        });
    }
}

template <typename ContextT, auto Fn,
          int32_t threads_per_invocation,
          int32_t items_per_invocation,
          typename ...ComponentTs>
uint32_t CustomParallelForNode<ContextT, Fn,
                               threads_per_invocation,
                               items_per_invocation,
                               ComponentTs...>::numInvocations()
{
    StateManager *state_mgr = mwGPU::getStateManager();
    int32_t num_entities = state_mgr->numMatchingEntities(query_ref_);
    return utils::divideRoundUp(num_entities, items_per_invocation);
}

template <typename ContextT, auto Fn,
          int32_t threads_per_invocation,
          int32_t items_per_invocation,
          typename ...ComponentTs>
TaskGraph::NodeID CustomParallelForNode<ContextT, Fn,
                                        threads_per_invocation,
                                        items_per_invocation,
                                        ComponentTs...>::addToGraph(
    TaskGraph::Builder &builder,
    Span<const TaskGraph::NodeID> dependencies)
{
    return builder.addDynamicCountNode<
        CustomParallelForNode<
            ContextT, Fn,
            threads_per_invocation,
            items_per_invocation,
            ComponentTs...>>(dependencies, threads_per_invocation);
}

template <typename ArchetypeT>
TaskGraph::NodeID ClearTmpNode<ArchetypeT>::addToGraph(
    TaskGraph::Builder &builder,
    Span<const TaskGraph::NodeID> dependencies)
{
    return ClearTmpNodeBase::addToGraph(builder, dependencies,
        TypeTracker::typeID<ArchetypeT>());
}

template <typename ArchetypeT, typename ComponentT>
TaskGraph::NodeID SortArchetypeNode<ArchetypeT, ComponentT>::addToGraph(
    TaskGraph::Builder &builder,
    Span<const TaskGraph::NodeID> dependencies)
{
    return SortArchetypeNodeBase::addToGraph(builder, dependencies,
        TypeTracker::typeID<ArchetypeT>(),
        TypeTracker::typeID<ComponentT>());
}

template <typename ArchetypeT>
TaskGraph::NodeID CompactArchetypeNode<ArchetypeT>::addToGraph(
    TaskGraph::Builder &builder,
    Span<const TaskGraph::NodeID> dependencies)
{
    auto sort_sys = builder.addToGraph<SortArchetypeNode<ArchetypeT, WorldID>>(
        dependencies);
    auto post_sort_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_sys});

    return post_sort_reset_tmp;
}

}
