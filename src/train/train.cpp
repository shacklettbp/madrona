#include <cstdint>
#include <iostream>
#include <array>

#include <madrona/train.hpp>

#include "cuda_utils.hpp"
#include "cpp_compile.hpp"

// Wrap this header in the gpuTrain namespace. This is a weird situation where
// the CPU job system headers are available but we need access to the GPU
// header in order to do initial setup.
namespace gpuTrain {
#include "gpu/madrona/job.hpp"
}

namespace madrona {

namespace ICfg {
static constexpr uint32_t numLaunchKernelThreads = 512;
static constexpr uint32_t numJobSystemKernelThreads = 1024;
}

using GPUJobManager = gpuTrain::madrona::JobManager;
using GPUJob = gpuTrain::madrona::Job;

struct GPUKernels {
    CUmodule mod;
    CUfunction initJobSystem;
    CUfunction runJobSystem;
    CUfunction queueUserInit;
    CUfunction queueUserRun;
};

struct GPUEngineState {
    GPUJobManager *jobSystemState;
};

struct TrainingExecutor::Impl {
    cudaStream_t cuStream;
    CUmodule cuModule;
    GPUJobManager *jobSystemState;
    CUgraphExec runGraph;
};

static HeapArray<char> makeEntryCode(const char *init_func,
                                     const char *run_func,
                                     const char *user_namespace)
{
    const char *entry_code_template = R"__(
#include <madrona/job.hpp>

// Forward declare user functions
namespace %s {
extern void %s(madrona::Context &ctx);
extern void %s(madrona::Context &ctx);
}

extern "C" __global__ void madronaTrainQueueUserInitKernel()
{
    madrona::Context ctx {};
    ctx.queueJob([](madrona::Context &ctx) {
        ::%s::%s(ctx);
    });
}

extern "C" __global__ void madronaTrainQueueUserRunKernel()
{
    madrona::Context ctx {};
    ctx.queueJob([](madrona::Context &ctx) {
        ::%s::%s(ctx);
    });
}
)__";

    HeapArray<char> entry_code(1 + strlen(entry_code_template) - 2 * 7 +
                               3 * strlen(user_namespace) +
                               2 * strlen(init_func) + 2 * strlen(run_func));
    
    snprintf(entry_code.data(), entry_code.size(),
             entry_code_template, user_namespace, init_func, run_func,
             user_namespace, init_func, user_namespace, run_func);

    return entry_code;
}

static CUmodule compileCode(const char **sources, uint32_t num_sources,
    const char **compile_flags, uint32_t num_compile_flags,
    const char *entry_code)
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

    return mod;
}

static GPUKernels buildKernels(const CompileConfig &cfg, uint32_t gpu_id)
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

    HeapArray<char> entry_code =
        makeEntryCode(cfg.initFunc, cfg.runFunc, cfg.userNamespace);

    GPUKernels gpu_kernels;
    gpu_kernels.mod = compileCode(all_cpp_files.data(),
        all_cpp_files.size(), compile_flags.data(), compile_flags.size(),
        entry_code.data());

    REQ_CU(cuModuleGetFunction(&gpu_kernels.initJobSystem, gpu_kernels.mod,
                               "madronaTrainInitializeJobSystemKernel"));
    REQ_CU(cuModuleGetFunction(&gpu_kernels.runJobSystem, gpu_kernels.mod,
                               "madronaTrainJobSystemKernel"));
    REQ_CU(cuModuleGetFunction(&gpu_kernels.queueUserInit, gpu_kernels.mod,
                               "madronaTrainQueueUserInitKernel"));
    REQ_CU(cuModuleGetFunction(&gpu_kernels.queueUserRun, gpu_kernels.mod,
                               "madronaTrainQueueUserRunKernel"));

    return gpu_kernels;
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

template <typename ...Ts>
HeapArray<void *> makeKernelArgBuffer(Ts ...args)
{
    uint32_t num_arg_bytes = 0;
    auto incrementArgSize = [&num_arg_bytes](auto v) {
        using T = decltype(v);
        num_arg_bytes = utils::roundUp(num_arg_bytes,
                                       (uint32_t)std::alignment_of_v<T>);
        num_arg_bytes += sizeof(T);
    };

    ( incrementArgSize(args), ... );

    uint32_t arg0_alignment = ( std::alignment_of_v<decltype(args)>, ... );

    uint32_t total_buf_size = sizeof(void *) * 5;
    uint32_t arg_size_ptr_offset =
        utils::roundUp(total_buf_size, (uint32_t)std::alignment_of_v<size_t>);

    total_buf_size = arg_size_ptr_offset + sizeof(size_t);

    uint32_t arg_ptr_offset = utils::roundUp(total_buf_size, arg0_alignment);

    total_buf_size = arg_ptr_offset + num_arg_bytes;

    HeapArray<void *> arg_buffer(utils::divideRoundUp(
            total_buf_size, (uint32_t)sizeof(void *)));

    size_t *arg_size_start = (size_t *)(
        (char *)arg_buffer.data() + arg_size_ptr_offset);

    new (arg_size_start) size_t(num_arg_bytes);

    void *arg_start = (char *)arg_buffer.data() + arg_ptr_offset;

    uint32_t cur_arg_offset = 0;
    auto copyArgs = [arg_start, &cur_arg_offset](auto v) {
        using T = decltype(v);

        cur_arg_offset =
            utils::roundUp(cur_arg_offset, (uint32_t)std::alignment_of_v<T>);

        memcpy((char *)arg_start + cur_arg_offset, &v, sizeof(T));

        cur_arg_offset += sizeof(T);
    };

    ( copyArgs(args), ... );

    arg_buffer[0] = CU_LAUNCH_PARAM_BUFFER_POINTER;
    arg_buffer[1] = arg_start;
    arg_buffer[2] = CU_LAUNCH_PARAM_BUFFER_SIZE;
    arg_buffer[3] = arg_size_start;
    arg_buffer[4] = CU_LAUNCH_PARAM_END;

    return arg_buffer;
}

static GPUEngineState initEngineAndUserState(uint32_t num_worlds,
                                             const GPUKernels &gpu_kernels,
                                             cudaStream_t strm)
{

    // Expand buffers based on number of parallel worlds
    (void)num_worlds;

    auto job_mgr = (GPUJobManager *)cu::allocGPU(sizeof(GPUJobManager));

    auto arg_buffer = makeKernelArgBuffer(job_mgr);

    auto launchKernel = [&](CUfunction f, uint32_t num_blocks,
                            uint32_t num_threads) {
        REQ_CU(cuLaunchKernel(f, num_blocks, 1, 1, num_threads, 1, 1,
                              0, strm, nullptr, arg_buffer.data()));
    };

    launchKernel(gpu_kernels.initJobSystem, 1, 1);
    // FIXME: this will break with more than 1024 worlds
    // Need to figure out proper way to pass in world index mapping for a
    // given thread
    launchKernel(gpu_kernels.queueUserInit, 1, num_worlds); 
    launchKernel(gpu_kernels.runJobSystem, 1,
                 ICfg::numJobSystemKernelThreads);
    launchKernel(gpu_kernels.queueUserRun, 1,
                 ICfg::numJobSystemKernelThreads);

    REQ_CUDA(cudaStreamSynchronize(strm));

    void *job_mgr_cache = job_mgr + sizeof(GPUJobManager) /* + FIXME */;

    REQ_CUDA(cudaMemcpy(job_mgr_cache, job_mgr,
                        utils::roundUp(sizeof(GPUJobManager),
                                       std::alignment_of_v<GPUJob>) +
                            sizeof(GPUJob),
                        cudaMemcpyDeviceToDevice));
}

static CUgraphExec makeRunGraph(void **arg_config,
                                CUfunction queue_run_kernel,
                                CUfunction job_sys_kernel,
                                uint32_t num_launch_blocks)
{
    CUgraph run_graph;
    REQ_CU(cuGraphCreate(&run_graph, 0));

    CUDA_KERNEL_NODE_PARAMS kernel_node_params {
        .func = queue_run_kernel,
        .gridDimX = num_launch_blocks,
        .gridDimY = 1,
        .gridDimZ = 1,
        .blockDimX = ICfg::numLaunchKernelThreads,
        .blockDimY = 1,
        .blockDimZ = 1,
        .sharedMemBytes = 0,
        .kernelParams = nullptr,
        .extra = arg_config,
    };

    CUgraphNode launch_node;
    REQ_CU(cuGraphAddKernelNode(&launch_node, run_graph,
        nullptr, 0, &kernel_node_params));

    kernel_node_params.func = job_sys_kernel;
    kernel_node_params.gridDimX = 1;
    kernel_node_params.blockDimX = ICfg::numJobSystemKernelThreads;

    CUgraphNode job_sys_node;
    REQ_CU(cuGraphAddKernelNode(&job_sys_node, run_graph,
        &launch_node, 1, &kernel_node_params));

    CUgraphExec run_graph_exec;
    REQ_CU(cuGraphInstantiate(&run_graph_exec, run_graph,
                              nullptr,  nullptr, 0));

    REQ_CU(cuGraphDestroy(run_graph));

    return run_graph_exec;
}

TrainingExecutor::TrainingExecutor(const TrainConfig &train_cfg,
                                   const CompileConfig &compile_cfg)
    : impl_(nullptr)
{
    auto strm = cu::makeStream();
    
    GPUKernels gpu_kernels = buildKernels(compile_cfg, train_cfg.gpuID);

    GPUEngineState eng_state = initEngineAndUserState(train_cfg.numWorlds,
                                                      gpu_kernels, strm);

    uint32_t num_launch_blocks = utils::divideRoundUp(train_cfg.numWorlds,
        ICfg::numLaunchKernelThreads);

    auto run_graph = makeRunGraph(arg_config.data(), gpu_kernels.queueUserRun,
                                  gpu_kernels.runJobSystem, num_launch_blocks);

    impl_ = std::unique_ptr<Impl>(new Impl {
        strm,
        gpu_kernels.mod,
        job_sys_state,
        run_graph,
    });
}

TrainingExecutor::~TrainingExecutor()
{
    REQ_CU(cuGraphExecDestroy(impl_->runGraph));
    REQ_CU(cuModuleUnload(impl_->cuModule));
    REQ_CUDA(cudaStreamDestroy(impl_->cuStream));
}

void TrainingExecutor::run()
{
    REQ_CU(cuGraphLaunch(impl_->runGraph, impl_->cuStream));
    REQ_CUDA(cudaStreamSynchronize(impl_->cuStream));
}

}
