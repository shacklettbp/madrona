#include <cstdint>
#include <iostream>
#include <array>

#include <string>

#include <madrona/mw_gpu.hpp>

#include "cuda_utils.hpp"
#include "cpp_compile.hpp"

// Wrap this header in the mwGPU namespace. This is a weird situation where
// the CPU job system headers are available but we need access to the GPU
// header in order to do initial setup.
namespace mwGPU {
using namespace madrona;
#include "device/include/madrona/mw_gpu/const.hpp"
}

namespace madrona {

namespace ICfg {
static constexpr uint32_t numJobSystemKernelThreads = 1024;
static constexpr uint32_t numEntryQueueThreads = 512;
}

using GPUImplConsts = mwGPU::madrona::mwGPU::GPUImplConsts;

struct GPUKernels {
    CUmodule mod;
    CUfunction computeGPUImplConsts;
    CUfunction init;
    CUfunction runJobSystem;
    CUfunction queueUserInit;
    CUfunction queueUserRun;
};

struct GPUEngineState {
    void *stateBuffer;
};

struct TrainingExecutor::Impl {
    cudaStream_t cuStream;
    CUmodule cuModule;
    GPUEngineState engineState; 
    CUgraphExec runGraph;
};

static void getUserEntries(const char *entry_class, CUmodule mod,
                           const char **compile_flags,
                           uint32_t num_compile_flags,
                           CUfunction *init_out, CUfunction *run_out)
{
    static const char mangle_code_postfix[] = R"__(
#include <cstdint>

namespace madrona { namespace mwGPU {

template <typename T> __global__ void submitInit(uint32_t) {}
template <typename T> __global__ void submitRun(uint32_t) {}

} }
)__";

    static const char init_template[] =
        "::madrona::mwGPU::submitInit<::";
    static const char run_template[] =
        "::madrona::mwGPU::submitRun<::";

    std::string_view entry_view(entry_class);

    // If user prefixed with ::, trim off as it will be added later
    if (entry_view[0] == ':' && entry_view[1] == ':') {
        entry_view = entry_view.substr(2);
    }

    std::string fwd_declare; 

    // Find all namespace separators
    int num_namespaces = 0;
    size_t prev_off = 0, off = 0;
    while ((off = entry_view.find("::", prev_off)) != std::string_view::npos) {
        auto ns_view = entry_view.substr(prev_off, off - prev_off);

        fwd_declare += "namespace ";
        fwd_declare += ns_view;
        fwd_declare += " { ";

        prev_off = off + 2;
        num_namespaces++;
    }

    auto class_view = entry_view.substr(prev_off);
    if (class_view.size() == 0) {
        FATAL("Invalid entry class name\n");
    }

    fwd_declare += "class ";
    fwd_declare += class_view;
    fwd_declare += "; ";

    for (int i = 0; i < num_namespaces; i++) {
        fwd_declare += "} ";
    }

    std::string mangle_code = std::move(fwd_declare);
    mangle_code += mangle_code_postfix;

    std::string init_name = init_template;
    init_name += entry_view;
    init_name += ">";

    std::string run_name = run_template;
    run_name += entry_view;
    run_name += ">";

    nvrtcProgram prog;
    REQ_NVRTC(nvrtcCreateProgram(&prog, mangle_code.c_str(), "mangle.cpp",
                                 0, nullptr, nullptr));

    REQ_NVRTC(nvrtcAddNameExpression(prog, init_name.c_str()));
    REQ_NVRTC(nvrtcAddNameExpression(prog, run_name.c_str()));

    REQ_NVRTC(nvrtcCompileProgram(prog, num_compile_flags, compile_flags));

    const char *init_lowered;
    REQ_NVRTC(nvrtcGetLoweredName(prog, init_name.c_str(), &init_lowered));
    const char *run_lowered;
    REQ_NVRTC(nvrtcGetLoweredName(prog, run_name.c_str(), &run_lowered));

    REQ_CU(cuModuleGetFunction(init_out, mod, init_lowered));
    REQ_CU(cuModuleGetFunction(run_out, mod, run_lowered));

    REQ_NVRTC(nvrtcDestroyProgram(&prog));
}

static CUmodule compileCode(const char **sources, uint32_t num_sources,
    const char **compile_flags, uint32_t num_compile_flags)
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
        MADRONA_MW_GPU_INTERNAL_CPP
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

    GPUKernels gpu_kernels;
    gpu_kernels.mod = compileCode(all_cpp_files.data(),
        all_cpp_files.size(), compile_flags.data(), compile_flags.size());

    REQ_CU(cuModuleGetFunction(&gpu_kernels.computeGPUImplConsts,
        gpu_kernels.mod, "madronaTrainComputeGPUImplConstantsKernel"));
    REQ_CU(cuModuleGetFunction(&gpu_kernels.init, gpu_kernels.mod,
                               "madronaTrainInitializeKernel"));
    REQ_CU(cuModuleGetFunction(&gpu_kernels.runJobSystem, gpu_kernels.mod,
                               "madronaTrainJobSystemKernel"));

    getUserEntries(cfg.entryName, gpu_kernels.mod, base_compile_flags.data(),
        base_compile_flags.size(), &gpu_kernels.queueUserInit,
        &gpu_kernels.queueUserRun);

    return gpu_kernels;
}

template <typename ...Ts>
HeapArray<void *> makeKernelArgBuffer(Ts ...args)
{
    if constexpr (sizeof...(args) == 0) {
        HeapArray<void *> arg_buffer(6);
        arg_buffer[0] = (void *)uintptr_t(0);
        arg_buffer[1] = CU_LAUNCH_PARAM_BUFFER_POINTER;
        arg_buffer[2] = nullptr;
        arg_buffer[3] = CU_LAUNCH_PARAM_BUFFER_SIZE;
        arg_buffer[4] = &arg_buffer[0];
        arg_buffer[5] = CU_LAUNCH_PARAM_END;

        return arg_buffer;
    } else {
        uint32_t num_arg_bytes = 0;
        auto incrementArgSize = [&num_arg_bytes](auto v) {
            using T = decltype(v);
            num_arg_bytes = utils::roundUp(num_arg_bytes,
                                           (uint32_t)std::alignment_of_v<T>);
            num_arg_bytes += sizeof(T);
        };

        ( incrementArgSize(args), ... );

        auto getArg0Align = [](auto arg0, auto ...) {
            return std::alignment_of_v<decltype(arg0)>;
        };

        uint32_t arg0_alignment = getArg0Align(args...);

        uint32_t total_buf_size = sizeof(void *) * 5;
        uint32_t arg_size_ptr_offset = utils::roundUp(total_buf_size,
            (uint32_t)std::alignment_of_v<size_t>);

        total_buf_size = arg_size_ptr_offset + sizeof(size_t);

        uint32_t arg_ptr_offset =
            utils::roundUp(total_buf_size, arg0_alignment);

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

            cur_arg_offset = utils::roundUp(cur_arg_offset,
                                            (uint32_t)std::alignment_of_v<T>);

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
}

static GPUEngineState initEngineAndUserState(uint32_t num_worlds,
                                             void *ctx_data,
                                             uint32_t num_ctx_data_bytes,
                                             uint32_t ctx_data_alignment,
                                             const GPUKernels &gpu_kernels,
                                             cudaStream_t strm)
{

    auto launchKernel = [strm](CUfunction f, uint32_t num_blocks,
                               uint32_t num_threads,
                               HeapArray<void *> &args) {
        REQ_CU(cuLaunchKernel(f, num_blocks, 1, 1, num_threads, 1, 1,
                              0, strm, nullptr, args.data()));
    };

    auto gpu_consts_readback = (GPUImplConsts *)cu::allocReadback(
        sizeof(GPUImplConsts));

    auto gpu_state_size_readback = (size_t *)cu::allocReadback(
        sizeof(size_t));

    auto compute_consts_args = makeKernelArgBuffer(num_worlds,
                                                   num_ctx_data_bytes,
                                                   ctx_data_alignment,
                                                   gpu_consts_readback,
                                                   gpu_state_size_readback);

    auto queue_args = makeKernelArgBuffer(num_worlds);

    auto no_args = makeKernelArgBuffer();

    launchKernel(gpu_kernels.computeGPUImplConsts, 1, 1,
                 compute_consts_args);

    REQ_CUDA(cudaStreamSynchronize(strm));

    auto gpu_state_buffer = cu::allocGPU(*gpu_state_size_readback);

    // The initial values of these pointers are equal to their offsets from
    // the base pointer. Now that we have the base pointer, write the
    // real pointer values.
    gpu_consts_readback->jobSystemAddr =
        (char *)gpu_consts_readback->jobSystemAddr +
        (uintptr_t)gpu_state_buffer;

    gpu_consts_readback->stateManagerAddr =
        (char *)gpu_consts_readback->stateManagerAddr +
        (uintptr_t)gpu_state_buffer;

    gpu_consts_readback->ctxDataAddr =
        (char *)gpu_consts_readback->ctxDataAddr +
        (uintptr_t)gpu_state_buffer;

    for (int i = 0; i < (int)num_worlds; i++) {
        cudaMemcpyAsync((char *)gpu_consts_readback->ctxDataAddr +
                            i * num_ctx_data_bytes,
                        ctx_data, num_ctx_data_bytes,
                        cudaMemcpyHostToDevice, strm);
    }

    CUdeviceptr job_sys_consts_addr;
    size_t job_sys_consts_size;
    REQ_CU(cuModuleGetGlobal(&job_sys_consts_addr, &job_sys_consts_size,
                             gpu_kernels.mod, "madronaTrainGPUImplConsts"));

    REQ_CU(cuMemcpyHtoD(job_sys_consts_addr, gpu_consts_readback,
                        job_sys_consts_size));

    launchKernel(gpu_kernels.init, 1, ICfg::numJobSystemKernelThreads,
                 no_args);
    
    uint32_t num_queue_blocks =
        utils::divideRoundUp(num_worlds, ICfg::numEntryQueueThreads);

    launchKernel(gpu_kernels.queueUserInit, num_queue_blocks,
                 ICfg::numEntryQueueThreads, queue_args); 

    launchKernel(gpu_kernels.runJobSystem, 1,
                 ICfg::numJobSystemKernelThreads, no_args);

    REQ_CUDA(cudaStreamSynchronize(strm));

    return GPUEngineState {
        gpu_state_buffer,
    };
}

static CUgraphExec makeRunGraph(CUfunction queue_run_kernel,
                                CUfunction job_sys_kernel,
                                uint32_t num_worlds)
{
    auto queue_args = makeKernelArgBuffer(num_worlds);
    auto no_args = makeKernelArgBuffer();

    uint32_t num_queue_blocks = utils::divideRoundUp(num_worlds,
        ICfg::numEntryQueueThreads);

    CUgraph run_graph;
    REQ_CU(cuGraphCreate(&run_graph, 0));

    CUDA_KERNEL_NODE_PARAMS kernel_node_params {
        .func = queue_run_kernel,
        .gridDimX = num_queue_blocks,
        .gridDimY = 1,
        .gridDimZ = 1,
        .blockDimX = ICfg::numEntryQueueThreads,
        .blockDimY = 1,
        .blockDimZ = 1,
        .sharedMemBytes = 0,
        .kernelParams = nullptr,
        .extra = queue_args.data(),
    };

    CUgraphNode queue_node;
    REQ_CU(cuGraphAddKernelNode(&queue_node, run_graph,
        nullptr, 0, &kernel_node_params));

    kernel_node_params.func = job_sys_kernel;
    kernel_node_params.gridDimX = 1;
    kernel_node_params.blockDimX = ICfg::numJobSystemKernelThreads;
    kernel_node_params.extra = no_args.data();

    CUgraphNode job_sys_node;
    REQ_CU(cuGraphAddKernelNode(&job_sys_node, run_graph,
        &queue_node, 1, &kernel_node_params));

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

    GPUEngineState eng_state = initEngineAndUserState(
        train_cfg.numWorlds, train_cfg.ctxData, train_cfg.numCtxDataBytes,
        train_cfg.ctxDataAlignment, gpu_kernels, strm);

    auto run_graph = makeRunGraph(gpu_kernels.queueUserRun,
                                  gpu_kernels.runJobSystem,
                                  train_cfg.numWorlds);

    impl_ = std::unique_ptr<Impl>(new Impl {
        strm,
        gpu_kernels.mod,
        eng_state,
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
