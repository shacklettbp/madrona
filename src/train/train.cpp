#include <cstdint>
#include <iostream>
#include <array>

#include <madrona/train.hpp>

#include "cuda_utils.hpp"
#include "cpp_compile.hpp"

// Wrap this header in the gpuTrain namespace. This is a weird situation where the
// CPU job system headers are available but we need access to the GPU header
// in order to do initial setup.
namespace gpuTrain {
#include "gpu/madrona/job.hpp"
}

namespace madrona {

using GPUJobQueue = gpuTrain::madrona::JobQueue;

static bool compileCode(const char **filenames, uint32_t num_filenames,
                        const char *init_job_name, uint32_t gpu_id,
                        CUfunction *init_job, CUfunction *job_system)
{
    auto searchForFunction = [](CUmodule mod, const char *fname) {
        CUfunction func_search;
        CUresult res = cuModuleGetFunction(&func_search, mod, fname);
        if (res == CUDA_SUCCESS) {
            return func_search;
        } else if (res == CUDA_ERROR_NOT_FOUND) {
            return (CUfunction)nullptr;
        } else {
            ERR_CU(res);
        }
    };

    *init_job = nullptr;
    *job_system = nullptr;
    for (int i = 0; i < (int)num_filenames; i++) {
        auto cubin = cu::compileToCUBIN(filenames[i], gpu_id, nullptr, 0);

        CUmodule mod;
        REQ_CU(cuModuleLoadData(&mod, cubin.data()));

        CUfunction init_search = searchForFunction(mod, init_job_name);
        CUfunction job_search = searchForFunction(mod, "jobSystemKernel");

        if (init_search != nullptr) {
            if (*init_job != nullptr) {
                FATAL("Multiple definitions of entry point: '%s'\n",
                      init_job_name);
            }
            *init_job = init_search;
        }

        if (job_search != nullptr) {
            if (*job_system != nullptr) {
                FATAL("Multiple definitions of 'jobSystem'\n");
            }
            *job_system = job_search;
        }
    }

    return *init_job != nullptr && *job_system != nullptr;
}

static void buildKernels(const char **added_filenames,
                         uint32_t num_added_filenames,
                         uint32_t gpu_id,
                         CUfunction *init_job, CUfunction *job_system)
{
    std::array internal_cpp_files {
        MADRONA_TRAIN_INTERNAL_CPP
    };

    HeapArray<const char *> all_cpp_files(internal_cpp_files.size() +
                                          num_added_filenames);
    memcpy(all_cpp_files.data(), internal_cpp_files.data(),
           sizeof(const char *) * internal_cpp_files.size());

    memcpy(all_cpp_files.data() + internal_cpp_files.size(),
           added_filenames,
           sizeof(const char *) * num_added_filenames);

    bool compile_success = compileCode(all_cpp_files.data(),
        all_cpp_files.size(), "initJob", gpu_id, init_job, job_system);

    if (!compile_success) {
        FATAL("Necessary entry points not found in provided C++ files");
    }
}

#if 0
__global__ void setInitialJobKernelAddress(JobQueue *job_queue)
{
    job_queue->jobs[0].fn = jobEntry<Fn>;
}

JobQueue *initJobSystem(cudaStream_t strm, Fn &&fn)
{
    JobQueue *job_queue = (JobQueue *)cu::allocGPU(sizeof(JobQueue));
    JobQueue *queue_staging = (JobQueue *)cu::allocStaging(sizeof(JobQueue));

    queue_staging->jobHead = 0;
    queue_staging->numWaitingJobs = 1;
    queue_staging->numOutstandingJobs = 0;

    setInitialJobKernelAddress<Fn><<<1, 1, 0, strm>>>(queue_staging);

    queue_staging->jobs[0].arg = &job_queue->jobData.buffer;

    new (&(queue_staging->jobData.buffer)[0]) Fn(std::forward<Fn>(fn));

    cu::cpyCPUToGPU(strm, job_queue, queue_staging, sizeof(JobQueue));
    REQ_CUDA(cudaStreamSynchronize(strm));

    cu::deallocCPU(queue_staging);

    return job_queue;
}
#endif

GPUJobQueue * initJobSystem(uint32_t num_worlds, cudaStream_t strm)
{
    // Expand buffers based on number of parallel worlds
    (void)num_worlds;

    auto job_queue = (GPUJobQueue *)cu::allocGPU(sizeof(GPUJobQueue));
    auto queue_staging = (GPUJobQueue *)cu::allocStaging(sizeof(GPUJobQueue));

    queue_staging->jobHead = 0;
    queue_staging->numWaitingJobs = 0;
    queue_staging->numOutstandingJobs = 0;

    cu::cpyCPUToGPU(strm, job_queue, queue_staging, sizeof(GPUJobQueue));
    REQ_CUDA(cudaStreamSynchronize(strm));

    cu::deallocCPU(queue_staging);

    return job_queue;
}

struct TrainingExecutor::Impl {
    cudaStream_t cuStream;
    CUfunction initJobKernel;
    CUfunction jobSystemKernel;
    gpuTrain::madrona::JobQueue *jobSystemState;
    uint32_t numLaunchBlocks;
};

TrainingExecutor::TrainingExecutor(const TrainConfig &cfg,
                                   Span<const char *> user_cpp_files)
    : impl_(nullptr)
{
    auto strm = cu::makeStream();
    
    CUfunction init_job_kernel, job_system_kernel;
    buildKernels(user_cpp_files.data(), user_cpp_files.size(), cfg.gpuID,
                 &init_job_kernel, &job_system_kernel);

    auto *job_queue = initJobSystem(cfg.numWorlds, strm);

    uint32_t num_launch_blocks = utils::divideRoundUp(cfg.numWorlds, 512u);

    impl_ = std::unique_ptr<Impl>(new Impl {
        strm,
        init_job_kernel,
        job_system_kernel,
        job_queue,
        num_launch_blocks,
    });
}

TrainingExecutor::~TrainingExecutor()
{}

void TrainingExecutor::run()
{
    REQ_CU(cuLaunchKernel(impl_->initJobKernel, impl_->numLaunchBlocks, 1, 1,
                          512, 1, 1, 0, impl_->cuStream,
                          (void **)&impl_->jobSystemState, nullptr));

    REQ_CU(cuLaunchKernel(impl_->jobSystemKernel, 1, 1, 1,
                          1024, 1, 1, 0, impl_->cuStream,
                          (void **)&impl_->jobSystemState, nullptr));

    REQ_CUDA(cudaStreamSynchronize(impl_->cuStream));
}

}
