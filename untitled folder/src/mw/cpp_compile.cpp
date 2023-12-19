/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include "cpp_compile.hpp"

#include <madrona/cuda_utils.hpp>

#include <array>
#include <fstream>

namespace madrona {
namespace cu {

CompileOutput jitCompileCPPSrc(const char *src,
                               const char *src_path,
                               const char **opt_compile_flags,
                               uint32_t num_opt_compile_flags,
                               const char **fast_compile_flags,
                               uint32_t num_fast_compile_flags,
                               bool ltoir_out)
{
    auto print_compile_log = [](nvrtcProgram prog) {
        // Retrieve log output
        size_t log_size = 0;
        REQ_NVRTC(nvrtcGetProgramLogSize(prog, &log_size));

        if (log_size > 1) {
            HeapArray<char> nvrtc_log(log_size);
            REQ_NVRTC(nvrtcGetProgramLog(prog, nvrtc_log.data()));
            fprintf(stderr, "%s\n\n", nvrtc_log.data());
        }

    };

    auto getLTOIR = [](nvrtcProgram prog) {
        size_t num_ltoir_bytes;
        REQ_NVRTC(nvrtcGetLTOIRSize(prog, &num_ltoir_bytes));

        HeapArray<char> ltoir_data(num_ltoir_bytes);

        REQ_NVRTC(nvrtcGetLTOIR(prog, ltoir_data.data()));

        return ltoir_data;
    };

    auto getPTX = [](nvrtcProgram prog) {
        size_t num_ptx_bytes;
        REQ_NVRTC(nvrtcGetPTXSize(prog, &num_ptx_bytes));

        HeapArray<char> ptx(num_ptx_bytes);
        REQ_NVRTC(nvrtcGetPTX(prog, ptx.data()));

        return ptx;
    };

    auto getCUBIN = [](nvrtcProgram prog) {
        size_t num_cubin_bytes;
        REQ_NVRTC(nvrtcGetCUBINSize(prog, &num_cubin_bytes));

        HeapArray<char> cubin_data(num_cubin_bytes);

        REQ_NVRTC(nvrtcGetCUBIN(prog, cubin_data.data()));

        return cubin_data;
    };

    auto ltoGetPTX = [&]() {
        nvrtcProgram fake_prog;
        REQ_NVRTC(nvrtcCreateProgram(&fake_prog, src, src_path, 0,
                                     nullptr, nullptr));

        nvrtcResult res = nvrtcCompileProgram(fake_prog,
            num_fast_compile_flags, fast_compile_flags);

        if (res != NVRTC_SUCCESS) {
            print_compile_log(fake_prog);
            ERR_NVRTC(res);
        }

        HeapArray<char> ptx = getPTX(fake_prog);

        REQ_NVRTC(nvrtcDestroyProgram(&fake_prog));

        return ptx;
    };

    nvrtcProgram prog;
    REQ_NVRTC(nvrtcCreateProgram(&prog, src, src_path, 0,
                                 nullptr, nullptr));

    nvrtcResult res = nvrtcCompileProgram(prog, num_opt_compile_flags,
        opt_compile_flags);

    print_compile_log(prog);
    if (res != NVRTC_SUCCESS) {
        ERR_NVRTC(res);
    }

    HeapArray<char> ptx = ltoir_out ? ltoGetPTX() : getPTX(prog);
    HeapArray<char> result = ltoir_out ? getLTOIR(prog) : getCUBIN(prog);

    REQ_NVRTC(nvrtcDestroyProgram(&prog));

    return CompileOutput {
        .outputPTX = std::move(ptx),
        .outputBinary = std::move(result),
    };
}

CompileOutput jitCompileCPPFile(const char *src_path,
                                  const char **opt_compile_flags,
                                  uint32_t num_opt_compile_flags,
                                  const char **fast_compile_flags,
                                  uint32_t num_fast_compile_flags,
                                  bool ltoir_out)
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
        opt_compile_flags, num_opt_compile_flags,
        fast_compile_flags, num_fast_compile_flags, ltoir_out);
}

}
}
