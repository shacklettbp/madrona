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
#include <madrona/batch_renderer.hpp>

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
static constexpr uint32_t numMegakernelThreads = 256;
static constexpr uint32_t numEntryQueueThreads = 512;
}

using GPUImplConsts = mwGPU::madrona::mwGPU::GPUImplConsts;

struct GPUKernels {
    CUmodule mod;
    CUfunction computeGPUImplConsts;
    CUfunction init;
    CUfunction megakernel;
    CUfunction queueUserInit;
    CUfunction queueUserRun;
};

struct GPUEngineState {
    render::BatchRenderer batchRenderer;
    void *stateBuffer;
    uint32_t *rendererInstanceCounts;
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
    CompileConfig::OptMode opt_mode, CompileConfig::Executor exec_mode)
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

    std::string megakernel_job_prefix = R"__(#include "megakernel_job_impl.inl"

extern "C" {

)__";

    std::string megakernel_taskgraph_prefix = R"__(#include "megakernel_taskgraph_impl.inl"

extern "C" {

)__";

    std::string megakernel_job_body = R"__(namespace madrona {
namespace mwGPU {

static __attribute__((always_inline)) inline void dispatch(
        uint32_t func_id,
        madrona::JobContainerBase *data,
        uint32_t *data_indices,
        uint32_t *invocation_offsets,
        uint32_t num_launches,
        uint32_t grid)
{
    switch (func_id) {
)__";

    std::string megakernel_taskgraph_body = R"__(namespace madrona {
namespace mwGPU {

static __attribute__((always_inline)) inline void dispatch(
        uint32_t func_id,
        mwGPU::EntryData &entry_data,
        uint32_t invocation_offset)
{
    switch (func_id) {
)__";

    std::string megakernel_prefix;
    std::string megakernel_body;
    std::string_view entry_prefix;
    std::string_view entry_postfix;
    std::string_view entry_params;
    std::string_view entry_args;
    std::string_view id_prefix;
    if (exec_mode == CompileConfig::Executor::JobSystem) {
        megakernel_prefix = megakernel_job_prefix;
        megakernel_body = megakernel_job_body;
        entry_prefix = ".weak .func _ZN7madrona5mwGPU8jobEntry";
        entry_postfix = "EvPNS_16JobContainerBaseEPj";
        entry_params = "(madrona::JobContainerBase *, uint32_t *, uint32_t *, uint32_t, uint32_t);\n";
        entry_args = "(data, data_indices, invocation_offsets, num_launches, grid);\n";
        id_prefix = "_ZN7madrona5mwGPU13JobFuncIDBase";
    } else if (exec_mode == CompileConfig::Executor::TaskGraph) {
        megakernel_prefix = megakernel_taskgraph_prefix;
        megakernel_body = megakernel_taskgraph_body;
        entry_prefix = ".weak .func _ZN7madrona5mwGPU9userEntry";
        entry_postfix = "EvRNS0_9EntryDataEi";
        entry_params = "(madrona::mwGPU::EntryData &, int32_t);\n";
        entry_args = "(entry_data, invocation_offset);\n";
        id_prefix = "_ZN7madrona5mwGPU14UserFuncIDBase";
    }

    uint32_t cur_func_id = 0;
    auto megakernelAddPTXEntries = [&megakernel_prefix,
        &megakernel_body, &cur_func_id, entry_prefix, entry_postfix,
        entry_params, entry_args, id_prefix](std::string_view ptx) {

        using namespace std::literals;
        using SizeT = std::string_view::size_type;

        constexpr SizeT start_skip = ".weak .func "sv.size();

        SizeT cur_pos = 0;
        SizeT found_pos;
        while ((found_pos = ptx.find(entry_prefix, cur_pos)) != ptx.npos) {
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
            megakernel_prefix += entry_params;

            SizeT postfix_start = mangled_fn.find(entry_postfix);
            assert(postfix_start != mangled_fn.npos);

            SizeT id_common_start = entry_prefix.size() - ".weak .func "sv.size();
            auto common = mangled_fn.substr(id_common_start,
                                            postfix_start - id_common_start);
            
            auto id_str = std::to_string(cur_func_id);

            megakernel_prefix += "uint32_t ";
            megakernel_prefix += id_prefix;
            megakernel_prefix += common;
            megakernel_prefix += "2idE = ";
            megakernel_prefix += id_str;
            megakernel_prefix += ";\n";

            megakernel_body += "        case ";
            megakernel_body += id_str;
            megakernel_body += ": {\n";
            megakernel_body += "            ";
            megakernel_body += mangled_fn;
            megakernel_body += entry_args;
            megakernel_body += "        } break;\n";

            cur_func_id++;
        }
    };

    for (int i = 0; i < (int)num_sources; i++) {
        printf("%s\n", sources[i]);
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

static GPUKernels buildKernels(const CompileConfig &cfg,
                               uint32_t gpu_id)
{
    using namespace std;

    array job_sys_cpp_files {
        MADRONA_MW_GPU_JOB_SYS_INTERNAL_CPP
    };

    array task_graph_cpp_files {
        MADRONA_MW_GPU_TASK_GRAPH_INTERNAL_CPP
    };

    uint32_t num_exec_srcs = 0;
    const char **exec_srcs = nullptr;
    if (cfg.execMode == CompileConfig::Executor::JobSystem) {
        num_exec_srcs = job_sys_cpp_files.size();
        exec_srcs = job_sys_cpp_files.data();
    } else if (cfg.execMode == CompileConfig::Executor::TaskGraph) {
        num_exec_srcs = task_graph_cpp_files.size();
        exec_srcs = task_graph_cpp_files.data();
    }
    size_t num_srcs = num_exec_srcs + cfg.userSources.size();

    HeapArray<const char *> all_cpp_files(num_srcs);
    memcpy(all_cpp_files.data(), exec_srcs,
           sizeof(const char *) * num_exec_srcs);

    memcpy(all_cpp_files.data() + num_exec_srcs,
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

    if (cfg.execMode == CompileConfig::Executor::JobSystem) {
        compile_flags.push_back("-DMARONA_MWGPU_JOB_SYSTEM=1");
    } else if (cfg.execMode == CompileConfig::Executor::TaskGraph) {
        compile_flags.push_back("-DMADRONA_MWGPU_TASKGRAPH=1");
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
        cfg.optMode, cfg.execMode);

    REQ_CU(cuModuleGetFunction(&gpu_kernels.computeGPUImplConsts,
        gpu_kernels.mod, "madronaMWGPUComputeConstants"));
    REQ_CU(cuModuleGetFunction(&gpu_kernels.megakernel, gpu_kernels.mod,
                               "madronaMWGPUMegakernel"));

    if (cfg.execMode == CompileConfig::Executor::JobSystem) {
        REQ_CU(cuModuleGetFunction(&gpu_kernels.init, gpu_kernels.mod,
                                   "madronaMWGPUInitialize"));
        getUserEntries(cfg.entryName, gpu_kernels.mod, compile_flags.data(),
            compile_flags.size(), &gpu_kernels.queueUserInit,
            &gpu_kernels.queueUserRun);
    } else if (cfg.execMode == CompileConfig::Executor::TaskGraph) {
        REQ_CU(cuModuleGetFunction(&gpu_kernels.init, gpu_kernels.mod,
                                   "madronaMWGPUInitialize"));
    }

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

static GPUEngineState initEngineAndUserState(int gpu_id,
                                             uint32_t num_worlds,
                                             uint32_t num_world_data_bytes,
                                             uint32_t world_data_alignment,
                                             void *world_init_ptr,
                                             uint32_t num_world_init_bytes,
                                             const GPUKernels &gpu_kernels,
                                             CompileConfig::Executor exec_mode,
                                             cudaStream_t strm)
{
    constexpr int64_t max_instances_per_world = 100;
    render::BatchRenderer batch_renderer({
        .gpuID = gpu_id,
        .renderWidth = 64,
        .renderHeight = 64,
        .numWorlds = num_worlds,
        .numViews = num_worlds,
        .maxInstancesPerWorld = max_instances_per_world,
        .maxObjects = 1000,
    });

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

    auto init_args = makeKernelArgBuffer(num_worlds, init_tmp_buffer);

    auto no_args = makeKernelArgBuffer();

    launchKernel(gpu_kernels.computeGPUImplConsts, 1, 1,
                 compute_consts_args);

    REQ_CUDA(cudaStreamSynchronize(strm));

    auto gpu_state_buffer = cu::allocGPU(*gpu_state_size_readback);
    cu::deallocCPU(gpu_state_size_readback);

    // The initial values of these pointers are equal to their offsets from
    // the base pointer. Now that we have the base pointer, write the
    // real pointer values.
    gpu_consts_readback->jobSystemAddr =
        (char *)gpu_consts_readback->jobSystemAddr +
        (uintptr_t)gpu_state_buffer;

    gpu_consts_readback->taskGraph =
        (char *)gpu_consts_readback->taskGraph +
        (uintptr_t)gpu_state_buffer;

    gpu_consts_readback->stateManagerAddr =
        (char *)gpu_consts_readback->stateManagerAddr +
        (uintptr_t)gpu_state_buffer;

    gpu_consts_readback->chunkAllocatorAddr =
        (char *)gpu_consts_readback->chunkAllocatorAddr +
        (uintptr_t)gpu_state_buffer;

    gpu_consts_readback->chunkAllocatorAddr =
        (char *)gpu_consts_readback->chunkBaseAddr +
        (uintptr_t)gpu_state_buffer;

    gpu_consts_readback->worldDataAddr =
        (char *)gpu_consts_readback->worldDataAddr +
        (uintptr_t)gpu_state_buffer;

    gpu_consts_readback->rendererASInstancesAddrs =
        (void **)batch_renderer.tlasInstancePtrs();

    uint32_t *instance_counts_host = (uint32_t *)
        cu::allocReadback(sizeof(uint32_t *) * (uint64_t)num_worlds);

    gpu_consts_readback->rendererInstanceCountsAddr = instance_counts_host;
    gpu_consts_readback->rendererBLASesAddr = batch_renderer.objectsBLASPtr();

    CUdeviceptr job_sys_consts_addr;
    size_t job_sys_consts_size;
    REQ_CU(cuModuleGetGlobal(&job_sys_consts_addr, &job_sys_consts_size,
                             gpu_kernels.mod, "madronaMWGPUConsts"));

    REQ_CU(cuMemcpyHtoD(job_sys_consts_addr, gpu_consts_readback,
                        job_sys_consts_size));

    if (exec_mode == CompileConfig::Executor::JobSystem) {
        launchKernel(gpu_kernels.init, 1, consts::numMegakernelThreads,
                     no_args);
    
        uint32_t num_queue_blocks =
            utils::divideRoundUp(num_worlds, consts::numEntryQueueThreads);

        launchKernel(gpu_kernels.queueUserInit, num_queue_blocks,
                     consts::numEntryQueueThreads, init_args); 

        launchKernel(gpu_kernels.megakernel, 1,
                     consts::numMegakernelThreads, no_args);
    } else if (exec_mode == CompileConfig::Executor::TaskGraph) {
        launchKernel(gpu_kernels.init, 1, 1, init_args);
    }

    REQ_CUDA(cudaStreamSynchronize(strm));

    cu::deallocGPU(init_tmp_buffer);

    return GPUEngineState {
        std::move(batch_renderer),
        gpu_state_buffer,
        instance_counts_host,
    };
}

static CUgraphExec makeJobSysRunGraph(CUfunction queue_run_kernel,
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
    kernel_node_params.blockDimX = consts::numMegakernelThreads;
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

static CUgraphExec makeTaskGraphRunGraph(CUfunction megakernel)
{
    auto no_args = makeKernelArgBuffer();
    CUgraph run_graph;
    REQ_CU(cuGraphCreate(&run_graph, 0));

    uint32_t num_megakernel_blocks = 82 * 4; // FIXME

    CUDA_KERNEL_NODE_PARAMS kernel_node_params {
        .func = megakernel,
        .gridDimX = num_megakernel_blocks,
        .gridDimY = 1,
        .gridDimZ = 1,
        .blockDimX = consts::numMegakernelThreads,
        .blockDimY = 1,
        .blockDimZ = 1,
        .sharedMemBytes = 0,
        .kernelParams = nullptr,
        .extra = no_args.data(),
    };

    CUgraphNode megakernel_node;
    REQ_CU(cuGraphAddKernelNode(&megakernel_node, run_graph,
        nullptr, 0, &kernel_node_params));

    CUgraphExec run_graph_exec;
    REQ_CU(cuGraphInstantiate(&run_graph_exec, run_graph,
                               nullptr, nullptr, 0));

    REQ_CU(cuGraphDestroy(run_graph));

    return run_graph_exec;
}

TrainingExecutor::TrainingExecutor(const StateConfig &state_cfg,
                                   const CompileConfig &compile_cfg)
    : impl_(nullptr)
{
    // FIXME size limit for device side malloc:
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 8ul*1024ul*1024ul*1024ul);

    auto strm = cu::makeStream();
    
    GPUKernels gpu_kernels = buildKernels(compile_cfg, state_cfg.gpuID);

    GPUEngineState eng_state = initEngineAndUserState(
        (int)state_cfg.gpuID, state_cfg.numWorlds, state_cfg.numWorldDataBytes,
        state_cfg.worldDataAlignment, state_cfg.worldInitPtr,
        state_cfg.numWorldInitBytes, gpu_kernels, 
        compile_cfg.execMode, strm);

    auto run_graph =
        compile_cfg.execMode == CompileConfig::Executor::JobSystem ?
            makeJobSysRunGraph(gpu_kernels.queueUserRun,
                               gpu_kernels.megakernel,
                               state_cfg.numWorlds) :
            makeTaskGraphRunGraph(gpu_kernels.megakernel);

    impl_ = std::unique_ptr<Impl>(new Impl {
        strm,
        gpu_kernels.mod,
        std::move(eng_state),
        run_graph,
    });

    std::cout << "Initialization finished" << std::endl;
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

    impl_->engineState.batchRenderer.render(
        impl_->engineState.rendererInstanceCounts);
}

}
