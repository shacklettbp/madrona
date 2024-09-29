#pragma once

namespace madrona {

template <typename ContextT, typename WorldT, typename ConfigT, typename InitT>
TaskGraphExecutor<ContextT, WorldT, ConfigT, InitT>::TaskGraphExecutor(
        const Config &cfg,
        const ConfigT &user_cfg,
        const InitT *user_inits,
        CountT num_taskgraphs)
    : ThreadPoolExecutor(cfg),
      contexts_(cfg.numWorlds),
      world_datas_(cfg.numWorlds),
      job_datas_((CountT)cfg.numWorlds * num_taskgraphs),
      jobs_((CountT)cfg.numWorlds * num_taskgraphs),
      num_taskgraphs_((uint32_t)num_taskgraphs)
{
    auto ecs_reg = getECSRegistry();
    WorldT::registerTypes(ecs_reg, user_cfg);

    HeapArray<TaskGraphManager> taskgraph_mgrs(cfg.numWorlds);

    auto ctx_init_cb = [&](const WorkerInit &worker_init,
                           CountT world_idx) -> Context & {
        WorldT *world_data_ptr = &world_datas_[world_idx];
        new (&contexts_[world_idx]) Context(world_data_ptr, worker_init);

        taskgraph_mgrs.emplace(world_idx, num_taskgraphs, worker_init);

        return contexts_[world_idx];
    };

    using CBPtrT = decltype(&ctx_init_cb);
    initializeContexts([](void *ptr_raw,
                          const WorkerInit &worker_init,
                          CountT world_idx) -> Context & {
        return (*(CBPtrT)ptr_raw)(worker_init, world_idx);
    }, &ctx_init_cb, cfg.numWorlds);

    for (CountT world_idx = 0; world_idx < (CountT)cfg.numWorlds;
         world_idx++) {
        world_datas_.emplace(world_idx, contexts_[world_idx],
                             user_cfg, user_inits[world_idx]);
    }

    HeapArray<HeapArray<TaskGraph>> built_graphs(cfg.numWorlds);
    for (CountT world_idx = 0; world_idx < (CountT)cfg.numWorlds;
         world_idx++) {
        WorldT::setupTasks(taskgraph_mgrs[world_idx], user_cfg);

        built_graphs.emplace(world_idx,
                             taskgraph_mgrs[world_idx].constructGraphs());
    }

    for (CountT taskgraph_idx = 0; taskgraph_idx < num_taskgraphs;
         taskgraph_idx++) {
        for (CountT world_idx = 0; world_idx < (CountT)cfg.numWorlds;
             world_idx++) {
            CountT job_idx = taskgraph_idx * cfg.numWorlds + world_idx;

            job_datas_.emplace(job_idx, JobData {
                .ctx = &contexts_[world_idx],
                .taskgraph = std::move(built_graphs[world_idx][taskgraph_idx]),
            });

            jobs_[job_idx].fn = [](void *ptr) {
                auto job_data = (JobData *)ptr;
                job_data->taskgraph.run(job_data->ctx);
            };
            jobs_[job_idx].data = &job_datas_[job_idx];
        }
    }

    initExport();
}

template <typename ContextT, typename WorldT, typename ConfigT, typename InitT>
template <EnumType EnumT>
void TaskGraphExecutor<ContextT, WorldT, ConfigT, InitT>::runTaskGraph(EnumT taskgraph_id)
{
    runTaskGraph(static_cast<uint32_t>(taskgraph_id));
}

template <typename ContextT, typename WorldT, typename ConfigT, typename InitT>
void TaskGraphExecutor<ContextT, WorldT, ConfigT, InitT>::runTaskGraph(uint32_t taskgraph_idx)
{
    CountT offset = taskgraph_idx * world_datas_.size();
    ThreadPoolExecutor::run(jobs_.data() + offset, world_datas_.size());
}

template <typename ContextT, typename WorldT, typename ConfigT, typename InitT>
void TaskGraphExecutor<ContextT, WorldT, ConfigT, InitT>::run()
{
    for (uint32_t i = 0; i < (uint32_t)num_taskgraphs_; i++) {
        runTaskGraph(i);
    }
}

template <typename ContextT, typename WorldT, typename ConfigT, typename InitT>
WorldT & TaskGraphExecutor<ContextT, WorldT, ConfigT, InitT>::getWorldData(
    CountT world_idx)
{
    return world_datas_[world_idx];
}

template <typename ContextT, typename WorldT, typename ConfigT, typename InitT>
ContextT & TaskGraphExecutor<ContextT, WorldT, ConfigT, InitT>::getWorldContext(
    CountT world_idx)
{
    return contexts_[world_idx];
}

}
