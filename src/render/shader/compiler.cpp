#include "compiler.hpp"
#include "utils.hpp"

#include <madrona/macros.hpp>

#include <memory>

// On clang on linux and mac need to either compile with -fms-exceptions
// define __EMULATE_UUID. The latter seems simpler and matches GCC
#if defined(MADRONA_LINUX) or defined(MADRONA_APPLE)
#define __EMULATE_UUID 1
#endif
#include <dxc/dxcapi.h>
#undef __EMULATE_UUID

#include <madrona/heap_array.hpp>

namespace madrona::render {

struct ShaderCompiler::Impl {
    CComPtr<IDxcUtils> dxcUtils;
    CComPtr<IDxcCompiler3> dxcCompiler;
}

static void checkDXC(HRESULT res, const char *msg, const char *file,
                     int line, const char *funcname)
{
    if (!DXC_FAILED(res)) return;

    const char *err_code;
#define ERR_CASE(val) \
    case E_##val: \
        err_code = #val; \
    break

    switch(res) {
        ERR_CASE(ABORT);
        ERR_CASE(ACCESSDENIED);
        ERR_CASE(BOUNDS);
        ERR_CASE(FAIL);
        ERR_CASE(HANDLE);
        ERR_CASE(INVALIDARG);
        ERR_CASE(NOINTERFACE);
        ERR_CASE(NOTIMPL);
        ERR_CASE(NOT_VALID_STATE);
        ERR_CASE(OUTOFMEMORY);
        ERR_CASE(POINTER);
        ERR_CASE(UNEXPECTED);
        default:
            err_code = "Unknown DXC error code";
            break;
    }
#undef ERR_CASE

    fatal(file, line, funcname, "%s: %s", msg, err_code);
}

#define REQ_DXC(expr, msg) \
    ::madrona::render::checkDXC((expr), msg, __FILE__, __LINE__,\
                                MADRONA_COMPILER_FUNCTION_NAME)

ShaderCompiler::ShaderCompiler()
    : impl_([]() {
        CComPtr<IDxcUtils> dxc_utils;
        REQ_DXC(DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&dxc_utils)),
                "Failed to initialize DxcUtils");

        CComPtr<IDxcCompiler3> dxc_compiler;
        REQ_DXC(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&dxc_compiler)),
                "Failed to initialize DxcCompiler");

        return std::unique_ptr<ShaderCompiler::Impl>(new ShaderCompiler::Impl {
            move(dxc_utils),
            move(dxc_compiler),
        });
    }())
{}

static HeapArray<uint32_t> hlslToSPV(
    IDxcUtils *dxc_utils,
    IDxcCompiler3 *dxc_compiler,
    const char *shader_path,
    const char *entry_point,
    ShaderStage stage,
    Span<const char *> include_dirs,
    Span<const char *> defines)
{
    const uint32_t src_cp = CP_UTF8;
    CComPtr<IDxcBlobEncoding> src_blob;
    REQ_DXC(dxc_utils->CreateBlobFromFile(shader_path, &src_cp, &src_blob),
            "Failed to load shader source blob");

#define TYPE_STR(type_str) (type_str L"_6_7")
    LPCWSTR dxc_type;
    switch (stage) {
        case ShaderStage::Vertex:
            dxc_type = TYPE_STR(L"vs");
            break;
        case ShaderStage::Fragment:
            dxc_type = TYPE_STR(L"ps");
            break;
        case ShaderStage::Compute:
            dxc_type = TYPE_STR(L"cs");
            break;
        case ShaderStage::Mesh:
            dxc_type = TYPE_STR(L"ms");
            break;
        case ShaderStage::Amplification:
            dxc_type = TYPE_STR(L"as");
            break;
        default:
            FATAL("Shader compilation: Unsupported shader stage type");
    }
#undef TYPE_STR

    static constexpr array fixed_args = {
        DXC_ARG_WARNINGS_ARE_ERRORS,
        DXC_ARG_DEBUG,
        DXC_ARG_PACK_MATRIX_COLUMN_MAJOR,
        L"-spirv",
        L"-fspv-target-env=vulkan1.3",
        L"-fspv-reflect",
    };

    // FIXME
    wstring_convert<codecvt_utf8<wchar_t>> wconv;
    wstring wentry = wconv.from_bytes(entry);

    HeapArray<LPCWSTR> dxc_args(
        4 + fixed_args.size() + 2 * num_defines + 2 * num_includes);

    int arg_offset = 0;
    auto setDXCArg = [&dxc_args](LPCWSTR arg) {
        dxc_args[arg_offset++] = arg;
    };

    setDXCArg(L"-E");
    setDXCArg(wentry.c_str());
    setDXCArg(L"-T");
    setDXCArg(dxc_type);

    for (int i = 0; i < fixed_args.size(); i++) {
        setDXCArg(fixed_args[i]);
    }

    vector<wstring> wdefines;
    wdefines.reserve(num_defines);

    for (int i = 0; i < num_defines; i++) {
        wdefines.emplace_back(wconv.from_bytes(defines[i]));
        setDXCArg(L"-D");
        setDXCArg(wdefines.back().c_str());
    }

    vector<wstring> wincs;
    wincs.reserve(include_dirs.size());

    for (int i = 0; i < num_includes; i++) {
        wincs.emplace_back(wconv.from_bytes(include_dirs[i]));
        setDXCArg(L"-I");
        setDXCArg(wincs.back().c_str());
    }

    DxcBuffer src_info;
    src_info.Ptr = src_blob->GetBufferPointer();
    src_info.Size = src_blob->GetBufferSize();
    src_info.Encoding = 0;

    CComPtr<IDxcResult> compile_result;
    REQ_DXC(dxc_compiler->Compile(&src_info, dxc_args.data(),
                                  (uint32_t)dxc_args.size(),
                                  nullptr,
                                  IID_PPV_ARGS(&compile_result)),
            "Failed to compile shader");

    CComPtr<IDxcBlobUtf8> compile_errors;
    compile_result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&compile_errors),
                              nullptr);

    if (compile_errors && compile_errors->GetStringLength() > 0) {
        FATAL("Compilation failed: %s", compile_errors->GetBufferPointer());
    }

    CComPtr<IDxcBlob> dxc_spv;
    compile_result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&dxc_spv),
                              nullptr);

    uint32_t num_shader_bytes = dxc_spv->GetBufferSize();
    assert(num_shader_bytes % 4 == 0);

    HeapArray<uint32_t> spv(num_shader_bytes / 4);
    memcpy(spv.data(), dxc_spv->GetBufferPointer(),
           num_shader_bytes);

    return spv;
}

SPIRVShader compile

}
