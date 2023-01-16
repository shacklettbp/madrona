#include <madrona/mw_cpu.hpp>
#include "../core/worker_init.hpp"

namespace madrona {

static CountT getNumCores()
{
    int os_num_threads = sysconf(_SC_NPROCESSORS_ONLN);

    if (os_num_threads == -1) {
        FATAL("Failed to get number of concurrent threads");
    }

    return os_num_threads;
}

static inline void pinThread(CountT worker_id)
{
    cpu_set_t cpuset;
    pthread_getaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

    const int max_threads = CPU_COUNT(&cpuset);

    CPU_ZERO(&cpuset);

    if (worker_id > max_threads) [[unlikely]] {
        FATAL("Tried setting thread affinity to %d when %d is max",
              worker_id, max_threads);
    }

    CPU_SET(worker_id, &cpuset);

    int res = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

    if (res != 0) {
        FATAL("Failed to set thread affinity to %d", worker_id);
    }
}

static Optional<render::BatchRenderer> makeRenderer(
    const ThreadPoolExecutor::Config &cfg)
{
    if (cfg.cameraMode == ThreadPoolExecutor::CameraMode::None) {
        return Optional<render::BatchRenderer>::none();
    }

    return Optional<render::BatchRenderer>::make(render::BatchRenderer::Config {
        .gpuID = cfg.renderGPUID,
        .renderWidth = cfg.renderWidth,
        .renderHeight = cfg.renderHeight,
        .numWorlds = cfg.numWorlds,
        .maxViewsPerWorld = cfg.maxViewsPerWorld,
        .maxInstancesPerWorld = cfg.maxInstancesPerWorld,
        .maxObjects = cfg.maxObjects,
        .cameraMode =
            cfg.cameraMode == ThreadPoolExecutor::CameraMode::Perspective ?
                render::BatchRenderer::CameraMode::Perspective :
                render::BatchRenderer::CameraMode::Lidar,
        .inputMode = render::BatchRenderer::InputMode::CPU,
    });
}

ThreadPoolExecutor::ThreadPoolExecutor(const Config &cfg)
    : workers_(cfg.numWorkers == 0 ? getNumCores() : cfg.numWorkers),
      worker_wakeup_(0),
      main_wakeup_(0),
      current_jobs_(nullptr),
      num_jobs_(0),
      next_job_(0),
      num_finished_(0),
      state_mgr_(cfg.numWorlds),
      state_caches_(cfg.numWorlds),
      export_ptrs_(cfg.numExportedBuffers),
      renderer_(makeRenderer(cfg))
{
    for (CountT i = 0; i < (CountT)cfg.numWorlds; i++) {
        new (&state_caches_[i]) StateCache();
    }

    for (CountT i = 0; i < workers_.size(); i++) {
        new (&workers_[i]) std::thread([this, i]() {
            workerThread(i);
        });
    }
}

ThreadPoolExecutor::~ThreadPoolExecutor()
{
    worker_wakeup_.store(-1, std::memory_order_release);
    worker_wakeup_.notify_all();

    for (CountT i = 0; i < workers_.size(); i++) {
        workers_[i].join();
    }
}

void ThreadPoolExecutor::run(Job *jobs, CountT num_jobs)
{
    current_jobs_ = jobs;
    num_jobs_ = uint32_t(num_jobs);
    next_job_.store(0, std::memory_order_relaxed);
    num_finished_.store(0, std::memory_order_relaxed);
    worker_wakeup_.store(1, std::memory_order_release);
    worker_wakeup_.notify_all();

    main_wakeup_.wait(0, std::memory_order_acquire);
    main_wakeup_.store(0, std::memory_order_relaxed);

    if (renderer_.has_value()) {
        char *hack = getenv("MADRONA_RENDER_NOOP");
        if (hack && hack[0] == '1') {
            return;
        }
        renderer_->render();
    }
}

CountT ThreadPoolExecutor::loadObjects(Span<const imp::SourceObject> objs)
{
    return renderer_->loadObjects(objs);
}

uint8_t * ThreadPoolExecutor::rgbObservations() const
{
    return renderer_->rgbPtr();
}

float * ThreadPoolExecutor::depthObservations() const
{
    return renderer_->depthPtr();
}

void * ThreadPoolExecutor::getExported(CountT slot) const
{
    return export_ptrs_[slot];
}

void ThreadPoolExecutor::ctxInit(void (*init_fn)(void *, const WorkerInit &),
                                 void *init_data, CountT world_idx)
{
    WorkerInit worker_init {
        &state_mgr_,
        &state_caches_[world_idx],
        uint32_t(world_idx),
    };

    init_fn(init_data, worker_init);

}

ECSRegistry ThreadPoolExecutor::getECSRegistry()
{
    return ECSRegistry(&state_mgr_, export_ptrs_.data());
}

Optional<render::RendererInterface> ThreadPoolExecutor::getRendererInterface()
{
    return renderer_.has_value() ?
        renderer_->getInterface() :
        Optional<render::RendererInterface>::none();
}

void ThreadPoolExecutor::workerThread(CountT worker_id)
{
    pinThread(worker_id);

    while (true) {
        worker_wakeup_.wait(0, std::memory_order_relaxed);
        int32_t ctrl = worker_wakeup_.load(std::memory_order_acquire);

        if (ctrl == 0) {
            continue;
        } else if (ctrl == -1) {
            break;
        }

        while (true) {
            uint32_t job_idx =
                next_job_.fetch_add(1, std::memory_order_relaxed);

            if (job_idx == num_jobs_) {
                worker_wakeup_.store(0, std::memory_order_relaxed);
            }

            assert(job_idx < num_jobs_ + 100);

            if (job_idx >= num_jobs_) {
                break;
            }

            current_jobs_[job_idx].fn(current_jobs_[job_idx].data);

            // This has to be acq_rel so the finishing thread has seen
            // all the other threads' effects
            uint32_t prev_finished =
                num_finished_.fetch_add(1, std::memory_order_acq_rel);

            if (prev_finished == num_jobs_ - 1) {
                main_wakeup_.store(1, std::memory_order_release);
                main_wakeup_.notify_one();
            }
        }
    }
}

}
