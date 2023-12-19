#pragma once

namespace madrona {

template <typename ContextT, typename WorldT, typename ConfigT, typename InitT>
TaskGraphExecutor<ContextT, WorldT, ConfigT, InitT>::TaskGraphExecutor(
        const Config &cfg,
        const ConfigT &user_cfg,
        const InitT *user_inits)
    : ThreadPoolExecutor(cfg),
      run_datas_(cfg.numWorlds),
      world_datas_(cfg.numWorlds),
      jobs_(cfg.numWorlds)
{
    auto ecs_reg = getECSRegistry();
    WorldT::registerTypes(ecs_reg, user_cfg);

    auto ctx_init_cb = [&](const WorkerInit &worker_init,
                           CountT world_idx) -> Context & {
        WorldT *world_data_ptr = &world_datas_[world_idx];

        auto *run_data = new (&run_datas_[world_idx]) RunData(
            world_data_ptr, user_cfg, worker_init);
        return run_data->ctx;
    };

    using CBPtrT = decltype(&ctx_init_cb);
    initializeContexts([](void *ptr_raw,
                          const WorkerInit &worker_init,
                          CountT world_idx) -> Context & {
        return (*(CBPtrT)ptr_raw)(worker_init, world_idx);
    }, &ctx_init_cb, cfg.numWorlds);

    for (CountT i = 0; i < (CountT)cfg.numWorlds; i++) {
        world_datas_.emplace(i, run_datas_[i].ctx, user_cfg, user_inits[i]);

        jobs_[i].fn = [](void *ptr) {
            stepWorld(ptr);
        };
        jobs_[i].data = &run_datas_[i];
    }

    initExport();
}

template <typename ContextT, typename WorldT, typename ConfigT, typename InitT>
void TaskGraphExecutor<ContextT, WorldT, ConfigT, InitT>::run()
{
    ThreadPoolExecutor::run(jobs_.data(), jobs_.size());
}

template <typename ContextT, typename WorldT, typename ConfigT, typename InitT>
WorldT & TaskGraphExecutor<ContextT, WorldT, ConfigT, InitT>::getWorldData(
    CountT world_idx)
{
    return world_datas_[world_idx];
}

template <typename ContextT, typename WorldT, typename ConfigT, typename InitT>
void TaskGraphExecutor<ContextT, WorldT, ConfigT, InitT>::stepWorld(
    void *data_raw)
{
    auto run_data = (RunData *)data_raw;
    run_data->taskgraph.run(&run_data->ctx);
}

template <typename ContextT, typename WorldT, typename ConfigT, typename InitT>
TaskGraphExecutor<ContextT, WorldT, ConfigT, InitT>::RunData::RunData(
        WorldT *world_data, const ConfigT &cfg, const WorkerInit &init)
    : ctx(world_data, init),
      taskgraph([&cfg, &init]() {
          TaskGraphBuilder builder(init);
          WorldT::setupTasks(builder, cfg);
          return builder.build();
      }())
{}

}
