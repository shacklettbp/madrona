/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include "cpp_compile.hpp"
#include "cuda_utils.hpp"

#include <array>
#include <fstream>

namespace madrona {
namespace cu {

HeapArray<char> jitCompileCPPSrc(const char *src,
                                 const char *src_path,
                                 const char **compile_flags,
                                 uint32_t num_compile_flags,
                                 bool nvvm_out)
{
    nvrtcProgram prog;
    REQ_NVRTC(nvrtcCreateProgram(&prog, src, src_path, 0,
                                 nullptr, nullptr));

    nvrtcResult res = nvrtcCompileProgram(prog, num_compile_flags,
        compile_flags);

    auto print_compile_log = [&prog]() {
        // Retrieve log output
        size_t log_size = 0;
        REQ_NVRTC(nvrtcGetProgramLogSize(prog, &log_size));

        if (log_size > 1) {
            HeapArray<char> nvrtc_log(log_size);
            REQ_NVRTC(nvrtcGetProgramLog(prog, nvrtc_log.data()));
            fprintf(stderr, "%s\n\n", nvrtc_log.data());
        }

    };

    print_compile_log();
    if (res != NVRTC_SUCCESS) {
        ERR_NVRTC(res);
    }

    auto produceNVVM = [&]() {
        size_t num_nvvm_bytes;
        REQ_NVRTC(nvrtcGetNVVMSize(prog, &num_nvvm_bytes));

        HeapArray<char> nvvm_data(num_nvvm_bytes);

        REQ_NVRTC(nvrtcGetNVVM(prog, nvvm_data.data()));

        return nvvm_data;
    };

    auto produceCUBIN = [&]() {
        size_t num_cubin_bytes;
        REQ_NVRTC(nvrtcGetCUBINSize(prog, &num_cubin_bytes));

        HeapArray<char> cubin_data(num_cubin_bytes);

        REQ_NVRTC(nvrtcGetCUBIN(prog, cubin_data.data()));

        return cubin_data;
    };

    HeapArray<char> result = nvvm_out ? produceNVVM() : produceCUBIN();

    REQ_NVRTC(nvrtcDestroyProgram(&prog));

    return result;
}

HeapArray<char> jitCompileCPPFile(const char *src_path,
                                  const char **compile_flags,
                                  uint32_t num_compile_flags,
                                  bool nvvm_out)
{
    using namespace std;

    ifstream src_file(src_path, ios::binary | ios::ate);
    size_t num_src_bytes = src_file.tellg();
    src_file.seekg(ios::beg);

    HeapArray<char> src(num_src_bytes + 1);
    src_file.read(src.data(), num_src_bytes);
    src_file.close();
    src[num_src_bytes] = '\0';

    return jitCompileCPPSrc(src.data(), src_path,
                            compile_flags, num_compile_flags, nvvm_out);
}

}
}
