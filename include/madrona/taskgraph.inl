#pragma once

namespace madrona {

template <typename NodeT, typename... Args> 
TaskGraph::TypedDataID<NodeT> TaskGraph::Builder::constructNodeData(
    Args && ...args)
{
    static_assert(sizeof(NodeT) <= maxNodeDataBytes);
    static_assert(alignof(NodeT) <= maxNodeDataBytes);

    CountT data_idx = node_datas_.uninit_back();
    new (&node_datas_[data_idx]) NodeT(std::forward<Args>(args)...);

    return TypedDataID<NodeT> {
        DataID { int32_t(data_idx) },
    };
}

template <auto fn, typename NodeT>
TaskGraph::NodeID TaskGraph::Builder::addNodeFn(
        TypedDataID<NodeT> data,
        Span<const NodeID> dependencies,
        Optional<NodeID> parent_node)
{
    return registerNode(uint32_t(data.id),
                        [](NodeBase *node_data, Context *ctx) {
                            std::invoke(fn, ((NodeT *)node_data), ctx);
                        },
                        dependencies,
                        parent_node);
}

template <typename NodeT, typename... Args>
TaskGraph::NodeID TaskGraph::Builder::addDefaultNode(
    Span<const NodeID> dependencies,
    Args && ...args)
{
    auto data_id = constructNodeData<NodeT>(
        std::forward<Args>(args)...);
    return addNodeFn<&NodeT::run>(data_id, dependencies,
                                  Optional<NodeID>::none());
}

template <typename NodeT>
NodeT & TaskGraph::Builder::getDataRef(TypedDataID<NodeT> data_id)
{
    return *(NodeT *)node_datas_[data_id.id].userData;
}

template <typename NodeT>
TaskGraph::NodeID TaskGraph::Builder::addToGraph(
    Span<const NodeID> dependencies)
{
    return NodeT::addToGraph(*ctx_, *this, dependencies);
}

template <typename ContextT, typename WorldT, typename... InitTs>
template <typename... Args>
TaskGraphExecutor<ContextT, WorldT, InitTs...>::TaskGraphExecutor(
        const Config &cfg,
        const Args * ...user_init_ptrs)
    : ThreadPoolExecutor(cfg.numWorlds, cfg.numWorkers),
      world_contexts_(cfg.numWorlds),
      jobs_(cfg.numWorlds)
{
    auto ecs_reg = getECSRegistry();
    WorldT::registerTypes(ecs_reg);

    for (CountT i = 0; i < (CountT)cfg.numWorlds; i++) {
        std::array<void *, sizeof...(InitTs)> init_ptrs {
            (void *)&user_init_ptrs[i] ...,
            nullptr
        };

        // FIXME: this is super ugly because WorkerInit
        // isn't available at the header level
        auto cb = [&](const WorkerInit &worker_init) {
            std::apply([&](auto ...ptrs) {
                new (&world_contexts_[i]) WorldContext(
                    worker_init, *(InitTs *)ptrs ...);
            }, init_ptrs);
        };

        using CBPtrT = decltype(&cb);

        auto wrapper = [](void *ptr_raw, const WorkerInit &worker_init) {
            (*(CBPtrT)ptr_raw)(worker_init);
        };

        ctxInit(wrapper, &cb, i);
        jobs_[i].fn = [](void *ptr) {
            stepWorld(ptr);
        };
        jobs_[i].data = &world_contexts_[i];
    }
}

template <typename ContextT, typename WorldT, typename... InitTs>
void TaskGraphExecutor<ContextT, WorldT, InitTs...>::run()
{
    ThreadPoolExecutor::run(jobs_.data(), jobs_.size());
}

template <typename ContextT, typename WorldT, typename... InitTs>
TaskGraphExecutor<ContextT, WorldT, InitTs...>::WorldContext::WorldContext(
        const WorkerInit &worker_init,
        const InitTs & ...world_inits)
    : ctx(&worldData, worker_init),
      worldData(ctx, world_inits...),
      taskgraph([this]() {
          TaskGraph::Builder builder(ctx);
          WorldT::setupTasks(builder);
          return builder.build();
      }())
{}

template <typename ContextT, typename WorldT, typename... InitTs>
void TaskGraphExecutor<ContextT, WorldT, InitTs...>::stepWorld(void *data_raw)
{
    auto world_ctx = (WorldContext *)data_raw;
    world_ctx->taskgraph.run(&world_ctx->ctx);
}

template <typename ContextT, auto Fn, typename ...ComponentTs>
ParallelForNode<ContextT, Fn, ComponentTs...>::ParallelForNode(Context &ctx)
    : query_(ctx.query<ComponentTs...>())
{}

template <typename ContextT, auto Fn, typename ...ComponentTs>
void ParallelForNode<ContextT, Fn, ComponentTs...>::run(Context *ctx_base)
{
    ContextT &ctx = *static_cast<ContextT *>(ctx_base);

    ctx.forEach(query_, [&](auto &...refs) {
        Fn(ctx, refs...);
    });
}

template <typename ContextT, auto Fn, typename ...ComponentTs>
TaskGraph::NodeID ParallelForNode<ContextT, Fn, ComponentTs...>::addToGraph(
    Context &ctx,
    TaskGraph::Builder &builder,
    Span<const TaskGraph::NodeID> dependencies)
{
    using NodeT = ParallelForNode<ContextT, Fn, ComponentTs...>;
    return builder.addDefaultNode<NodeT>(dependencies, ctx);
}

void ResetTmpAllocNode::run(Context *ctx)
{
    ctx->resetTmpAlloc();
}

template <typename ArchetypeT>
void ClearTmpNode<ArchetypeT>::run(Context *ctx)
{
    ctx->clearArchetype<ArchetypeT>();
}

template <typename ArchetypeT>
TaskGraph::NodeID ClearTmpNode<ArchetypeT>::addToGraph(
    Context &,
    TaskGraph::Builder &builder,
    Span<const TaskGraph::NodeID> dependencies)
{
    return builder.addDefaultNode<ClearTmpNode>(dependencies);
}


}
