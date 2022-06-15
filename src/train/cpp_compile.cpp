#include "cpp_compile.hpp"
#include "cuda_utils.hpp"

#include <array>
#include <fstream>

using namespace std;

namespace madrona {
namespace cu {

HeapArray<char> compileToCUBIN(const char *src_path, int gpu_id,
                               const char **extra_options,
                               uint32_t num_extra_options)
{
    ifstream src_file(src_path, ios::binary | ios::ate);
    size_t num_src_bytes = src_file.tellg();
    src_file.seekg(ios::beg);

    HeapArray<char> src(num_src_bytes + 1);
    src_file.read(src.data(), num_src_bytes);
    src_file.close();
    src[num_src_bytes] = '\0';

    nvrtcProgram prog;
    REQ_NVRTC(nvrtcCreateProgram(&prog, src.data(), src_path, 0,
                                 nullptr, nullptr));

    // Compute architecture string for this GPU
    cudaDeviceProp dev_props;
    REQ_CUDA(cudaGetDeviceProperties(&dev_props, gpu_id));
    string arch_str = "sm_" + to_string(dev_props.major) + to_string(dev_props.minor);

    array const_nvrtc_opts {
        MADRONA_NVRTC_OPTIONS
        "-arch", arch_str.c_str(),
        "--device-debug",
        "--extra-device-vectorization",
    };

    HeapArray<const char *> nvrtc_options(
        const_nvrtc_opts.size() + num_extra_options);
    memcpy(nvrtc_options.data(), const_nvrtc_opts.data(),
           sizeof(const char *) * const_nvrtc_opts.size());

    for (int i = 0; i < (int)num_extra_options; i++) {
        nvrtc_options[const_nvrtc_opts.size() + i] = extra_options[i];
    }

    nvrtcResult res = nvrtcCompileProgram(prog, nvrtc_options.size(),
        nvrtc_options.data());

    auto print_compile_log = [&prog]() {
        // Retrieve log output
        size_t log_size = 0;
        REQ_NVRTC(nvrtcGetProgramLogSize(prog, &log_size));

        if (log_size > 1) {
            HeapArray<char> nvrtc_log(log_size);
            REQ_NVRTC(nvrtcGetProgramLog(prog, nvrtc_log.data()));
            printf("%s\n", nvrtc_log.data());
        }

    };

    print_compile_log();
    if (res != NVRTC_SUCCESS) {
        ERR_NVRTC(res);
    }

    size_t num_cubin_bytes;
    REQ_NVRTC(nvrtcGetCUBINSize(prog, &num_cubin_bytes));

    HeapArray<char> cubin_data(num_cubin_bytes);

    REQ_NVRTC(nvrtcGetCUBIN(prog, cubin_data.data()));

    REQ_NVRTC(nvrtcDestroyProgram(&prog));

    return cubin_data;
}

}
}
