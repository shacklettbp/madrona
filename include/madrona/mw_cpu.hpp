#pragma once

#include <madrona/taskgraph.hpp>
#include <madrona/batch_renderer.hpp>

namespace madrona {

class ThreadPoolExecutor {
public:
    enum class CameraMode : uint32_t {
        Perspective,
        Lidar,
        None,
    };

    struct Config {
        uint32_t numWorlds;
        uint32_t maxViewsPerWorld;
        uint32_t maxInstancesPerWorld;
        uint32_t renderWidth;
        uint32_t renderHeight;
        uint32_t maxObjects;
        uint32_t numExportedBuffers;
        CameraMode cameraMode;
        int32_t renderGPUID;
        uint32_t numWorkers = 0;
    };

    struct Job {
        void (*fn)(void *);
        void *data;
    };

    ThreadPoolExecutor(const Config &cfg);
    ~ThreadPoolExecutor();
    void run(Job *jobs, CountT num_jobs);

    CountT loadObjects(Span<const imp::SourceObject> objs);

    uint8_t * rgbObservations() const;
    float * depthObservations() const;

    void * getExported(CountT slot) const;

protected:
    void ctxInit(void (*init_fn)(void *, const WorkerInit &),
                 void *init_data, CountT world_idx);

    ECSRegistry getECSRegistry();

    Optional<render::RendererInterface> getRendererInterface();

private:
    void workerThread(CountT worker_id);

    HeapArray<std::thread> workers_;
    alignas(MADRONA_CACHE_LINE) std::atomic_int32_t worker_wakeup_;
    alignas(MADRONA_CACHE_LINE) std::atomic_int32_t main_wakeup_;
    Job *current_jobs_;
    uint32_t num_jobs_;
    alignas(MADRONA_CACHE_LINE) std::atomic_uint32_t next_job_;
    alignas(MADRONA_CACHE_LINE) std::atomic_uint32_t num_finished_;
    StateManager state_mgr_;
    HeapArray<StateCache> state_caches_;
    HeapArray<void *> export_ptrs_;
    Optional<render::BatchRenderer> renderer_;
};

template <typename ContextT, typename WorldT, typename ConfigT,
          typename... InitTs>
class TaskGraphExecutor : private ThreadPoolExecutor {
public:
    template <typename... Args>
    TaskGraphExecutor(const Config &cfg,
                      const ConfigT &user_cfg,
                      const Args * ... user_init_ptrs);

    inline void run();

    using ThreadPoolExecutor::loadObjects;
    using ThreadPoolExecutor::getExported;
    using ThreadPoolExecutor::rgbObservations;
    using ThreadPoolExecutor::depthObservations;

private:
    struct WorldContext {
        ContextT ctx;
        WorldT worldData;
        TaskGraph taskgraph;

        inline WorldContext(const WorkerInit &worker_init,
                            const ConfigT &user_cfg,
                            const InitTs & ...world_inits);
                            
    };

    static inline void stepWorld(void *data_raw);

    HeapArray<WorldContext> world_contexts_;
    HeapArray<Job> jobs_;
};

}

#include "mw_cpu.inl"
