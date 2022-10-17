/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include <cstdint>
#include <iostream>
#include <array>

#include <fstream>
#include <string>

#include <madrona/mw_gpu.hpp>
#include <madrona/dyn_array.hpp>

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

namespace consts {
static constexpr uint32_t numJobSystemKernelThreads = 512;
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

template <typename T> __global__ void submitInit(uint32_t, void *) {}
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
    const char **compile_flags, uint32_t num_compile_flags,
    const char **fast_compile_flags, uint32_t num_fast_compile_flags,
    CompileConfig::OptMode opt_mode)
{
    static std::array<char, 1024 * 1024> linker_info_log;
    static std::array<char, 1024 * 1024> linker_error_log;

    DynArray<CUjit_option> linker_options {
        CU_JIT_INFO_LOG_BUFFER,
        CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_ERROR_LOG_BUFFER,
        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_LOG_VERBOSE,
    };

    DynArray<void *> linker_option_values {
        (void *)linker_info_log.data(),
        (void *)linker_info_log.size(),
        (void *)linker_error_log.data(),
        (void *)linker_error_log.size(),
        (void *)1, /* Verbose */
    };

    CUjitInputType linker_input_type;
    if (opt_mode == CompileConfig::OptMode::LTO) {
        linker_options.push_back(CU_JIT_LTO);
        linker_option_values.push_back((void *)1);
        linker_options.push_back(CU_JIT_FTZ);
        linker_option_values.push_back((void *)1);
        linker_options.push_back(CU_JIT_PREC_DIV);
        linker_option_values.push_back((void *)0);
        linker_options.push_back(CU_JIT_PREC_SQRT);
        linker_option_values.push_back((void *)0);
        linker_options.push_back(CU_JIT_FMA);
        linker_option_values.push_back((void *)1);
        linker_options.push_back(CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES);
        linker_option_values.push_back((void *)1);

        linker_input_type = CU_JIT_INPUT_NVVM;
    } else {
        linker_input_type = CU_JIT_INPUT_CUBIN;
    }

    if (opt_mode == CompileConfig::OptMode::Debug) {
        linker_options.push_back(CU_JIT_GENERATE_DEBUG_INFO);
        linker_option_values.push_back((void *)1);
        linker_options.push_back(CU_JIT_OPTIMIZATION_LEVEL);
        linker_option_values.push_back((void *)0);
    } else {
        linker_options.push_back(CU_JIT_GENERATE_LINE_INFO);
        linker_option_values.push_back((void *)1);
        linker_options.push_back(CU_JIT_OPTIMIZATION_LEVEL);
        linker_option_values.push_back((void *)4);
    }

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

    auto addToLinker = [&](const HeapArray<char> &cubin, const char *name) {
        checkLinker(cuLinkAddData(linker, linker_input_type,
            (char *)cubin.data(), cubin.size(), name, 0, nullptr, nullptr));
    };

    std::string megakernel_prefix = R"__(#include "megakernel_impl.inl"

extern "C" {

)__";

    std::string megakernel_body = R"__(static __attribute__((always_inline)) inline void dispatch(uint32_t func_id, madrona::JobContainerBase *data, uint32_t *data_indices, uint32_t *invocation_offsets, uint32_t num_launches, uint32_t grid)
{
    switch (func_id) {
)__";

    uint32_t cur_func_id = 0;
    auto megakernelAddPTXEntries = [&megakernel_prefix, &megakernel_body,
                                    &cur_func_id](std::string_view ptx) {
        using namespace std::literals;
        using SizeT = std::string_view::size_type;

        auto prefix = ".weak .func _ZN7madrona5mwGPU8jobEntry"sv;
        auto postfix = "EvPNS_16JobContainerBaseEPj"sv;
        constexpr SizeT start_skip = ".weak .func "sv.size();

        SizeT cur_pos = 0;
        SizeT found_pos;
        while ((found_pos = ptx.find(prefix, cur_pos)) != ptx.npos) {
            SizeT start_pos = found_pos + start_skip;
            SizeT paren_pos = ptx.find('(', start_pos);
            SizeT endline_pos = ptx.find('\n', start_pos);
            cur_pos = paren_pos;

            // This is a forward declaration so skip it
            if (paren_pos > endline_pos) {
                continue;
            }

            auto mangled_fn = ptx.substr(start_pos, paren_pos - start_pos);
            megakernel_prefix += "void "sv;
            megakernel_prefix += mangled_fn;
            megakernel_prefix += "(madrona::JobContainerBase *, uint32_t *, uint32_t *, uint32_t, uint32_t);\n"sv;

            SizeT postfix_start = mangled_fn.find(postfix);
            assert(postfix_start != mangled_fn.npos);

            SizeT id_common_start = prefix.size() - ".weak .func "sv.size();
            auto common = mangled_fn.substr(id_common_start,
                                            postfix_start - id_common_start);
            
            auto id_str = std::to_string(cur_func_id);

            megakernel_prefix += "uint32_t ";
            megakernel_prefix += "_ZN7madrona5mwGPU13JobFuncIDBase"sv;
            megakernel_prefix += common;
            megakernel_prefix += "2idE = ";
            megakernel_prefix += id_str;
            megakernel_prefix += ";\n";

            megakernel_body += "        case ";
            megakernel_body += id_str;
            megakernel_body += ": {\n";
            megakernel_body += "            ";
            megakernel_body += mangled_fn;
            megakernel_body += "(data, data_indices, invocation_offsets, num_launches, grid);\n";
            megakernel_body += "        } break;\n";

            cur_func_id++;
        }
    };

    for (int i = 0; i < (int)num_sources; i++) {
        auto [ptx, bytecode] = cu::jitCompileCPPFile(sources[i],
            compile_flags, num_compile_flags,
            fast_compile_flags, num_fast_compile_flags,
            opt_mode == CompileConfig::OptMode::LTO);

        megakernelAddPTXEntries(std::string_view(ptx.data(), ptx.size()));
        addToLinker(bytecode, sources[i]);
    }

    megakernel_prefix += R"__(
}
)__";

    megakernel_body += R"__(        default:
            __builtin_unreachable();
    }
}
)__";

    std::string megakernel =
        std::move(megakernel_prefix) + std::move(megakernel_body);

    std::string fake_megakernel_cpp_path =
        std::string(MADRONA_MW_GPU_DEVICE_SRC_DIR) + "/megakernel.cpp";

    auto compiled_megakernel = cu::jitCompileCPPSrc(megakernel.c_str(),
        fake_megakernel_cpp_path.c_str(), compile_flags, num_compile_flags,
        fast_compile_flags, num_fast_compile_flags,
        opt_mode == CompileConfig::OptMode::LTO);
    addToLinker(compiled_megakernel.outputBinary, "megakernel.cpp");

    void *linked_cubin;
    size_t cubin_size;
    checkLinker(cuLinkComplete(linker, &linked_cubin, &cubin_size));

    std::ofstream cubin_out("/tmp/t.cubin", std::ios::binary);
    cubin_out.write((char *)linked_cubin, cubin_size);

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

    DynArray<const char *> fast_compile_flags {
        MADRONA_NVRTC_OPTIONS
        "-arch", arch_str.c_str(),
    };

    for (const char *user_flag : cfg.userCompileFlags) {
        fast_compile_flags.push_back(user_flag);
    }

    DynArray<const char *> compile_flags(fast_compile_flags.size());
    for (const char *flag : fast_compile_flags) {
        compile_flags.push_back(flag);
    }

    // No way to disable optimizations in nvrtc besides enabling debug mode
    fast_compile_flags.push_back("-G");

    if (cfg.optMode == CompileConfig::OptMode::Debug) {
        compile_flags.push_back("--device-debug");
    } else {
        compile_flags.push_back("-dopt=on");
        compile_flags.push_back("--extra-device-vectorization");
        compile_flags.push_back("-lineinfo");
    }

    if (cfg.optMode == CompileConfig::OptMode::LTO) {
        compile_flags.push_back("-dlto");
        compile_flags.push_back("-DMADRONA_MWGPU_LTO_MODE=1");
    }

    for (const char *src : all_cpp_files) {
        cout << src << endl;
    }

    for (const char *flag : compile_flags) {
        cout << flag << endl;
    }

    GPUKernels gpu_kernels;
    gpu_kernels.mod = compileCode(all_cpp_files.data(),
        all_cpp_files.size(), compile_flags.data(), compile_flags.size(),
        fast_compile_flags.data(), fast_compile_flags.size(),
        cfg.optMode);

    REQ_CU(cuModuleGetFunction(&gpu_kernels.computeGPUImplConsts,
        gpu_kernels.mod, "madronaTrainComputeGPUImplConstantsKernel"));
    REQ_CU(cuModuleGetFunction(&gpu_kernels.init, gpu_kernels.mod,
                               "madronaTrainInitializeKernel"));
    REQ_CU(cuModuleGetFunction(&gpu_kernels.runJobSystem, gpu_kernels.mod,
                               "madronaTrainJobSystemKernel"));

    getUserEntries(cfg.entryName, gpu_kernels.mod, compile_flags.data(),
        compile_flags.size(), &gpu_kernels.queueUserInit,
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
                                             uint32_t num_world_data_bytes,
                                             uint32_t world_data_alignment,
                                             void *world_init_ptr,
                                             uint32_t num_world_init_bytes,
                                             const GPUKernels &gpu_kernels,
                                             cudaStream_t strm)
{

    auto launchKernel = [strm](CUfunction f, uint32_t num_blocks,
                               uint32_t num_threads,
                               HeapArray<void *> &args) {
        REQ_CU(cuLaunchKernel(f, num_blocks, 1, 1, num_threads, 1, 1,
                              0, strm, nullptr, args.data()));
    };

    uint64_t num_init_bytes =
        (uint64_t)num_world_init_bytes * (uint64_t)num_worlds;
    auto init_tmp_buffer = cu::allocGPU(num_init_bytes);
    cudaMemcpyAsync(init_tmp_buffer, world_init_ptr,
                    num_init_bytes, cudaMemcpyHostToDevice, strm);

    auto gpu_consts_readback = (GPUImplConsts *)cu::allocReadback(
        sizeof(GPUImplConsts));

    auto gpu_state_size_readback = (size_t *)cu::allocReadback(
        sizeof(size_t));

    auto compute_consts_args = makeKernelArgBuffer(num_worlds,
                                                   num_world_data_bytes,
                                                   world_data_alignment,
                                                   gpu_consts_readback,
                                                   gpu_state_size_readback);

    auto queue_args = makeKernelArgBuffer(num_worlds, init_tmp_buffer);

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

    gpu_consts_readback->worldDataAddr =
        (char *)gpu_consts_readback->worldDataAddr +
        (uintptr_t)gpu_state_buffer;

    CUdeviceptr job_sys_consts_addr;
    size_t job_sys_consts_size;
    REQ_CU(cuModuleGetGlobal(&job_sys_consts_addr, &job_sys_consts_size,
                             gpu_kernels.mod, "madronaTrainGPUImplConsts"));

    REQ_CU(cuMemcpyHtoD(job_sys_consts_addr, gpu_consts_readback,
                        job_sys_consts_size));

    launchKernel(gpu_kernels.init, 1, consts::numJobSystemKernelThreads,
                 no_args);
    
    uint32_t num_queue_blocks =
        utils::divideRoundUp(num_worlds, consts::numEntryQueueThreads);

    launchKernel(gpu_kernels.queueUserInit, num_queue_blocks,
                 consts::numEntryQueueThreads, queue_args); 

    launchKernel(gpu_kernels.runJobSystem, 1,
                 consts::numJobSystemKernelThreads, no_args);

    REQ_CUDA(cudaStreamSynchronize(strm));

    cu::deallocGPU(init_tmp_buffer);

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
        consts::numEntryQueueThreads);

    CUgraph run_graph;
    REQ_CU(cuGraphCreate(&run_graph, 0));

    CUDA_KERNEL_NODE_PARAMS kernel_node_params {
        .func = queue_run_kernel,
        .gridDimX = num_queue_blocks,
        .gridDimY = 1,
        .gridDimZ = 1,
        .blockDimX = consts::numEntryQueueThreads,
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
    kernel_node_params.blockDimX = consts::numJobSystemKernelThreads;
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

TrainingExecutor::TrainingExecutor(const StateConfig &state_cfg,
                                   const CompileConfig &compile_cfg)
    : impl_(nullptr)
{
    auto strm = cu::makeStream();
    
    GPUKernels gpu_kernels = buildKernels(compile_cfg, state_cfg.gpuID);

    GPUEngineState eng_state = initEngineAndUserState(
        state_cfg.numWorlds, state_cfg.numWorldDataBytes,
        state_cfg.worldDataAlignment, state_cfg.worldInitPtr,
        state_cfg.numWorldInitBytes, gpu_kernels, strm);

    auto run_graph = makeRunGraph(gpu_kernels.queueUserRun,
                                  gpu_kernels.runJobSystem,
                                  state_cfg.numWorlds);

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
