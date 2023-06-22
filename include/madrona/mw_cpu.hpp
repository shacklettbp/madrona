#pragma once

#include <madrona/taskgraph.hpp>
#include <madrona/render/mw.hpp>
#include <madrona/importer.hpp>

namespace madrona {

class ThreadPoolExecutor {
public:
    struct Config {
        uint32_t numWorlds;
        uint32_t numExportedBuffers;
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

    CountT loadObjects(Span<const imp::SourceObject> objs);

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

template <typename ContextT, typename WorldT, typename ConfigT, typename InitT>
class TaskGraphExecutor : private ThreadPoolExecutor {
public:
    TaskGraphExecutor(
        const Config &cfg,
        const ConfigT &user_cfg,
        const InitT *user_inits);

    inline void run();

    using ThreadPoolExecutor::getExported;

    inline WorldT & getWorldData(CountT world_idx);

private:
    struct RunData {
        ContextT ctx;
        TaskGraph taskgraph;

        inline RunData(WorldT *world_data, const ConfigT &cfg,
                       const WorkerInit &worker_init);
    };

    static inline void stepWorld(void *data_raw);

    HeapArray<RunData> run_datas_;
    HeapArray<WorldT> world_datas_;
    HeapArray<Job> jobs_;
};

}

#include "mw_cpu.inl"
