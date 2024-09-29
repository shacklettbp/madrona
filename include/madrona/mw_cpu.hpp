/*
 * Copyright 2021-2023 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <madrona/taskgraph_builder.hpp>
#include <madrona/importer.hpp>
#include <madrona/registry.hpp>

namespace madrona {

// Base class for TaskGraphExecutor below, don't use directly
class ThreadPoolExecutor {
public:
    struct Config {
        // Batch size for the backend
        uint32_t numWorlds;
        // Number of exported ECS components
        uint32_t numExportedBuffers;
        // Number of worker threads
        uint32_t numWorkers = 0;
    };

    struct Job {
        void (*fn)(void *);
        void *data;
    };

    ThreadPoolExecutor(const Config &cfg);
    ThreadPoolExecutor(ThreadPoolExecutor &&o);

    ~ThreadPoolExecutor();
    void run(Job *jobs, CountT num_jobs);

    // Get the base pointer of the component data exported with
    // ECSRegister::exportColumn
    void * getExported(CountT slot) const;

protected:
    void initializeContexts(
        Context & (*init_fn)(void *, const WorkerInit &, CountT),
        void *init_data, CountT num_worlds);

    ECSRegistry getECSRegistry();

    void initExport();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// The TaskGraphExecutor class is the entry point for the CPU backend.
// Use as follows:
//   using MyCPUBackend = TaskGraphExecutor<
//       MyContextSubclass, MyPerWorldState, MyConfig, MyPerWorldInit>;
//
//   MyCPUBackend backend({
//      .numWorlds = 1024,
//      .numExportedBuffers = 5, // Make sure this is set correctly!
//      .numWorkers = 0, // (Autodetect number of CPU cores)
//   }, MyConfig {}, my_world_inits);
//
//   backend.run(); // Take one step
//
// The above code will initialize the simulation state with 
// 1024 copies of the MyPerWorldState class, passing MyConfig and the 
// appropriate my_world_inits reference to the MyPerWorldState constructor
template <typename ContextT, typename WorldT, typename ConfigT, typename InitT>
class TaskGraphExecutor : private ThreadPoolExecutor {
public:
    TaskGraphExecutor(
        const Config &cfg,
        const ConfigT &user_cfg,
        const InitT *user_inits,
        CountT num_taskgraphs);

    // Run one invocation of the task graph across all worlds (one step)
    template <EnumType EnumT>
    inline void runTaskGraph(EnumT taskgraph_id);

    inline void runTaskGraph(uint32_t taskgraph_idx);

    inline void run();

    // Get the base pointer of the component data exported with
    // ECSRegister::exportColumn
    using ThreadPoolExecutor::getExported;

    // Get a reference to the per world data class
    inline WorldT & getWorldData(CountT world_idx);

    inline ContextT & getWorldContext(CountT idx);

private:
    struct JobData {
        Context *ctx;
        TaskGraph taskgraph;
    };

    HeapArray<ContextT> contexts_;
    HeapArray<WorldT> world_datas_;
    HeapArray<JobData> job_datas_;
    HeapArray<Job> jobs_;
    uint32_t num_taskgraphs_;
};

}

#include "mw_cpu.inl"
