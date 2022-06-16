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

namespace InternalConfig {
static constexpr uint32_t numLaunchKernelThreads = 512;
static constexpr uint32_t numJobSystemKernelThreads = 1024;
}

using GPUJobQueue = gpuTrain::madrona::JobQueue;

static HeapArray<char> makeEntryCode(const char *entry_func,
                                     const char *entry_namespace)
{
    const char *entry_code_template = R"__(
namespace madrona {
struct JobQueue;
extern void jobSystem(JobQueue *);
}

namespace %s {
extern void %s(madrona::JobQueue *job);
}

extern "C" __global__ void launchKernel(madrona::JobQueue *job_queue)
{
    %s::%s(job_queue);
}

extern "C" __global__ void jobSystemKernel(madrona::JobQueue *job_queue)
{
    madrona::jobSystem(job_queue);
}
)__";

    HeapArray<char> entry_code(1 + strlen(entry_code_template) - 2 * 4 +
                               2 * (strlen(entry_func) + 
                                    strlen(entry_namespace)));
    
    snprintf(entry_code.data(), entry_code.size(),
             entry_code_template, entry_namespace, entry_func,
             entry_namespace, entry_func);

    return entry_code;
}

static void compileCode(const char **sources, uint32_t num_sources,
                        const char **compile_flags, uint32_t num_compile_flags,
                        const char *entry_code,
                        CUfunction *launch_kernel, CUfunction *job_sys_kernel)
{
    static std::array<char, 4096> linker_info_log;
    static std::array<char, 4096> linker_error_log;

    std::array linker_options {
        CU_JIT_INFO_LOG_BUFFER,
        CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_ERROR_LOG_BUFFER,
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_GENERATE_DEBUG_INFO,
    };

    std::array linker_option_values {
        (void *)linker_info_log.data(),
        (void *)linker_info_log.size(),
        (void *)linker_error_log.data(),
        (void *)linker_error_log.size(),
        (void *)1,
    };

    CUlinkState linker;
    REQ_CU(cuLinkCreate(linker_options.size(), linker_options.data(),
                        linker_option_values.data(), &linker));

    auto checkLinker = [&linker_option_values](CUresult res) {
        if (res != CUDA_SUCCESS) {
            fprintf(stderr, "CUDA linking Failed!\n");

            if ((uintptr_t)linker_option_values[1] > 0) {
                fprintf(stderr, "%s\n", linker_info_log.data());
            }

            if ((uintptr_t)linker_option_values[3] > 0) {
                fprintf(stderr, "%s\n", linker_error_log.data());
            }

            fprintf(stderr, "\n");

            ERR_CU(res);
        }
    };

    checkLinker(cuLinkAddFile(linker, CU_JIT_INPUT_LIBRARY,
                              MADRONA_CUDADEVRT_PATH,
                              0, nullptr, nullptr));

    auto addToLinker = [checkLinker, linker](HeapArray<char> &&cubin,
                                             const char *name) {
        checkLinker(cuLinkAddData(linker, CU_JIT_INPUT_CUBIN, cubin.data(),
                             cubin.size(), name, 0, nullptr, nullptr));
    };

    addToLinker(cu::compileSrcToCUBIN(entry_code, "generated_launch.cpp",
                                      compile_flags, num_compile_flags),
                "generated_launch.cpp");

    for (int i = 0; i < (int)num_sources; i++) {
        addToLinker(cu::compileFileToCUBIN(sources[i], compile_flags,
                                           num_compile_flags),
                    sources[i]);
    }

    void *linked_cubin;
    checkLinker(cuLinkComplete(linker, &linked_cubin, nullptr));

    if ((uintptr_t)linker_option_values[1] > 0) {
        printf("CUDA linking info:\n%s\n", linker_info_log.data());
    }

    CUmodule mod;
    REQ_CU(cuModuleLoadData(&mod, linked_cubin));

    REQ_CU(cuLinkDestroy(linker));

    REQ_CU(cuModuleGetFunction(launch_kernel, mod, "launchKernel"));
    REQ_CU(cuModuleGetFunction(job_sys_kernel, mod, "jobSystemKernel"));
}

static void buildKernels(const CompileConfig &cfg, uint32_t gpu_id,
                         CUfunction *init_job, CUfunction *job_system)
{
    using namespace std;

    array internal_cpp_files {
        MADRONA_TRAIN_INTERNAL_CPP
    };

    HeapArray<const char *> all_cpp_files(internal_cpp_files.size() +
                                          cfg.userSources.size());
    memcpy(all_cpp_files.data(), internal_cpp_files.data(),
           sizeof(const char *) * internal_cpp_files.size());

    memcpy(all_cpp_files.data() + internal_cpp_files.size(),
           cfg.userSources.data(),
           sizeof(const char *) * cfg.userSources.size());

    // Compute architecture string for this GPU
    cudaDeviceProp dev_props;
    REQ_CUDA(cudaGetDeviceProperties(&dev_props, gpu_id));
    string arch_str = "sm_" + to_string(dev_props.major) +
        to_string(dev_props.minor);

    array base_compile_flags {
        MADRONA_NVRTC_OPTIONS
        "-arch", arch_str.c_str(),
        "--device-debug",
        "--extra-device-vectorization",
    };

    HeapArray<const char *> compile_flags(
        base_compile_flags.size() + cfg.userCompileFlags.size());
    memcpy(compile_flags.data(), base_compile_flags.data(),
           sizeof(const char *) * base_compile_flags.size());

    memcpy(compile_flags.data() + base_compile_flags.size(),
           cfg.userCompileFlags.data(),
           sizeof(const char *) * cfg.userCompileFlags.size());

    for (const char *src : all_cpp_files) {
        cout << src << endl;
    }

    for (const char *flag : compile_flags) {
        cout << flag << endl;
    }

    compileCode(all_cpp_files.data(),
        all_cpp_files.size(), compile_flags.data(), compile_flags.size(),
        makeEntryCode(cfg.entryFunc, cfg.entryNamespace).data(),
        init_job, job_system);
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

static GPUJobQueue * initJobSystem(uint32_t num_worlds, cudaStream_t strm)
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

static CUgraphExec makeRunGraph(GPUJobQueue *job_sys_state,
                                CUfunction launch_kernel,
                                CUfunction job_sys_kernel,
                                uint32_t num_launch_blocks)
{
    // Assumes that the launch kernel and job system kernel have the same
    // arguments
    
    std::array<char, sizeof(uint64_t)> kernel_arg_buffer;
    memcpy(kernel_arg_buffer.data(), &job_sys_state, sizeof(uint64_t));

    size_t arg_buffer_size = kernel_arg_buffer.size();
    std::array launch_config {
        CU_LAUNCH_PARAM_BUFFER_POINTER, (void *)kernel_arg_buffer.data(),
        CU_LAUNCH_PARAM_BUFFER_SIZE, (void *)&arg_buffer_size,
        CU_LAUNCH_PARAM_END,
    };

    CUgraph run_graph;
    REQ_CU(cuGraphCreate(&run_graph, 0));

    CUDA_KERNEL_NODE_PARAMS kernel_node_params {
        .func = launch_kernel,
        .gridDimX = num_launch_blocks,
        .gridDimY = 1,
        .gridDimZ = 1,
        .blockDimX = InternalConfig::numLaunchKernelThreads,
        .blockDimY = 1,
        .blockDimZ = 1,
        .sharedMemBytes = 0,
        .kernelParams = nullptr,
        .extra = launch_config.data(),
    };

    CUgraphNode launch_node;
    REQ_CU(cuGraphAddKernelNode(&launch_node, run_graph,
        nullptr, 0, &kernel_node_params));

    kernel_node_params.func = job_sys_kernel;
    kernel_node_params.gridDimX = 1;
    kernel_node_params.blockDimX = InternalConfig::numJobSystemKernelThreads;

    CUgraphNode job_sys_node;
    REQ_CU(cuGraphAddKernelNode(&job_sys_node, run_graph,
        &launch_node, 1, &kernel_node_params));

    CUgraphExec run_graph_exec;
    REQ_CU(cuGraphInstantiate(&run_graph_exec, run_graph,
                              nullptr,  nullptr, 0));

    REQ_CU(cuGraphDestroy(run_graph));

    return run_graph_exec;
}

struct TrainingExecutor::Impl {
    cudaStream_t cuStream;
    gpuTrain::madrona::JobQueue *jobSystemState;
    CUgraphExec runGraph;
};

TrainingExecutor::TrainingExecutor(const TrainConfig &train_cfg,
                                   const CompileConfig &compile_cfg)
    : impl_(nullptr)
{
    auto strm = cu::makeStream();
    
    CUfunction launch_kernel, job_system_kernel;
    buildKernels(compile_cfg, train_cfg.gpuID,
                 &launch_kernel, &job_system_kernel);

    auto *job_sys_state = initJobSystem(train_cfg.numWorlds, strm);

    uint32_t num_launch_blocks = utils::divideRoundUp(train_cfg.numWorlds,
        InternalConfig::numLaunchKernelThreads);

    auto run_graph = makeRunGraph(job_sys_state, launch_kernel,
                                  job_system_kernel, num_launch_blocks);

    impl_ = std::unique_ptr<Impl>(new Impl {
        strm,
        job_sys_state,
        run_graph,
    });
}

TrainingExecutor::~TrainingExecutor()
{
    REQ_CU(cuGraphExecDestroy(impl_->runGraph));
    REQ_CUDA(cudaStreamDestroy(impl_->cuStream));
}

void TrainingExecutor::run()
{
    REQ_CU(cuGraphLaunch(impl_->runGraph, impl_->cuStream));
    REQ_CUDA(cudaStreamSynchronize(impl_->cuStream));
}

}
