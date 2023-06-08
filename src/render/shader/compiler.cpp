#include "compiler.hpp"

#include <madrona/macros.hpp>

#include <cstdlib>
#include <memory>
#include <codecvt>

// On clang on linux and mac need to either compile with -fms-exceptions
// define __EMULATE_UUID. The latter seems simpler and matches GCC
#if defined(MADRONA_LINUX) or defined(MADRONA_APPLE)
#define __EMULATE_UUID 1
#endif
#include <dxc/dxcapi.h>
#undef __EMULATE_UUID

#include <spirv_reflect.h>

#include <madrona/heap_array.hpp>
#include <madrona/dyn_array.hpp>

namespace madrona::render {

struct ShaderCompiler::Impl {
    CComPtr<IDxcUtils> dxcUtils;
    CComPtr<IDxcCompiler3> dxcCompiler;
};

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
            std::move(dxc_utils),
            std::move(dxc_compiler),
        });
    }())
{}

static void inline checkSPVReflect(SpvReflectResult res,
    const char *file, int line, const char *funcname)
{
    if (res == SPV_REFLECT_RESULT_SUCCESS) return;

    const char *err_str = nullptr;

#define ERR_CASE(val)              \
    case SPV_REFLECT_RESULT_##val: \
        err_str = #val;              \
        break

    switch (res) {
        ERR_CASE(NOT_READY);
        ERR_CASE(ERROR_PARSE_FAILED);
        ERR_CASE(ERROR_ALLOC_FAILED);
        ERR_CASE(ERROR_RANGE_EXCEEDED);
        ERR_CASE(ERROR_NULL_POINTER);
        ERR_CASE(ERROR_INTERNAL_ERROR);
        ERR_CASE(ERROR_COUNT_MISMATCH);
        ERR_CASE(ERROR_ELEMENT_NOT_FOUND);
        ERR_CASE(ERROR_SPIRV_INVALID_CODE_SIZE);
        ERR_CASE(ERROR_SPIRV_INVALID_MAGIC_NUMBER);
        ERR_CASE(ERROR_SPIRV_UNEXPECTED_EOF);
        ERR_CASE(ERROR_SPIRV_INVALID_ID_REFERENCE);
        ERR_CASE(ERROR_SPIRV_SET_NUMBER_OVERFLOW);
        ERR_CASE(ERROR_SPIRV_INVALID_STORAGE_CLASS);
        ERR_CASE(ERROR_SPIRV_RECURSION);
        ERR_CASE(ERROR_SPIRV_INVALID_INSTRUCTION);
        ERR_CASE(ERROR_SPIRV_UNEXPECTED_BLOCK_DATA);
        ERR_CASE(ERROR_SPIRV_INVALID_BLOCK_MEMBER_REFERENCE);
        ERR_CASE(ERROR_SPIRV_INVALID_ENTRY_POINT);
        ERR_CASE(ERROR_SPIRV_INVALID_EXECUTION_MODE);
        default:
            err_str = "Unknown SPIRV-Reflect error";
            break;
    }
#undef ERR_CASE

    fatal(file, line, funcname, "%s", err_str);
}

#define REQ_SPV_RFL(expr) checkSPVReflect((expr), __FILE__, __LINE__,\
                                          MADRONA_COMPILER_FUNCTION_NAME)

static HeapArray<wchar_t> toWide(const char *str)
{
    HeapArray<wchar_t> out(strlen(str) + 1);
    size_t num_conv = mbstowcs(out.data(), str, out.size());
    if ((CountT)num_conv != out.size() - 1) {
        FATAL("Wide character conversion failed");
    }

    return out;
}

static CComPtr<IDxcBlobEncoding> loadFileToDxcBlob(
    IDxcUtils *dxc_utils,
    const char *shader_path)
{
    HeapArray<wchar_t> lshader_path = toWide(shader_path);

    uint32_t src_cp = CP_UTF8;
    CComPtr<IDxcBlobEncoding> blob;
    REQ_DXC(dxc_utils->LoadFile(lshader_path.data(), &src_cp, &blob),
            "Failed to load shader file");

    return blob;
}

static HeapArray<uint32_t> hlslToSPV(
    LPVOID src_shader_buffer,
    SIZE_T src_shader_len,
    IDxcCompiler3 *dxc_compiler,
    Span<const char *> include_dirs,
    Span<const char *> defines)
{
#if 0
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
#endif

    DynArray<LPCWSTR> dxc_args {
        DXC_ARG_WARNINGS_ARE_ERRORS,
        DXC_ARG_DEBUG,
        DXC_ARG_PACK_MATRIX_COLUMN_MAJOR,
        L"-spirv",
        L"-fspv-target-env=vulkan1.3",
        L"-fspv-reflect",
    };

    //dxc_args.push_bacj(L"-E");
    //dxc_args.push_back(wentry.c_str());

    //setDXCArg(L"-T");
    //setDXCArg(dxc_type);

    DynArray<HeapArray<wchar_t>> wdefines(defines.size());

    for (CountT i = 0; i < defines.size(); i++) {
        wdefines.emplace_back(toWide(defines[i]));
        dxc_args.push_back(L"-D");
        dxc_args.push_back(wdefines.back().data());
    }

    DynArray<HeapArray<wchar_t>> wincs(include_dirs.size());

    for (CountT i = 0; i < include_dirs.size(); i++) {
        wincs.emplace_back(toWide(include_dirs[i]));
        dxc_args.push_back(L"-I");
        dxc_args.push_back(wincs.back().data());
    }

    DxcBuffer src_info;
    src_info.Ptr = src_shader_buffer;
    src_info.Size = src_shader_len;
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

static refl::Stage convertSPVReflectStage(
    SpvReflectShaderStageFlagBits stage)
{
    using namespace refl;

    switch (stage) {
        case SPV_REFLECT_SHADER_STAGE_VERTEX_BIT: return Stage::Vertex;
        case SPV_REFLECT_SHADER_STAGE_FRAGMENT_BIT: return Stage::Fragment;
        case SPV_REFLECT_SHADER_STAGE_COMPUTE_BIT: return Stage::Compute;
        case SPV_REFLECT_SHADER_STAGE_TASK_BIT_EXT: return Stage::Amplification;
        case SPV_REFLECT_SHADER_STAGE_MESH_BIT_EXT: return Stage::Mesh;
        default:
            FATAL("Unsupported SPIRV shader stage");
    }
}

static refl::BindingType convertSPVReflectBindingType(
    SpvReflectDescriptorType type_in)
{
    using namespace refl;

    switch (type_in) {
        case SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLER:
            return BindingType::Sampler;
        case SPV_REFLECT_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            return BindingType::Texture;
        case SPV_REFLECT_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            return BindingType::UniformBuffer;
        case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            return BindingType::StorageBuffer;
        default:
            FATAL("Unsupported SPIRV descriptor type");
    }
}

static refl::SPIRV buildSPIRVReflectionData(
    const void *spv_data,
    CountT num_spv_bytes)
{
    using namespace refl;

    SpvReflectShaderModule rfl_mod;
    REQ_SPV_RFL(spvReflectCreateShaderModule2(SPV_REFLECT_MODULE_FLAG_NO_COPY,
        num_spv_bytes, spv_data, &rfl_mod));

    HeapArray<EntryPoint> entry_points(rfl_mod.entry_point_count);
    DynArray<DynArray<Binding>> tmp_desc_sets(0);
    for (CountT i = 0; i < entry_points.size(); i++) {
        const auto &spv_rfl_entry = rfl_mod.entry_points[i];

        Stage stage = convertSPVReflectStage(spv_rfl_entry.shader_stage);
        entry_points[i] = EntryPoint {
            .name = spv_rfl_entry.name,
            .stage = stage,
        };

        for (uint32_t j = 0; j < spv_rfl_entry.descriptor_set_count; j++) {
            const SpvReflectDescriptorSet &spv_desc_set =
                spv_rfl_entry.descriptor_sets[j];

            if (spv_desc_set.set >= tmp_desc_sets.size()) {
                tmp_desc_sets.resize(spv_desc_set.set + 1, [](auto *ptr) {
                    new (ptr) DynArray<Binding>(0);
                });
            }

            for (uint32_t k = 0; k < spv_desc_set.binding_count; k++) {
                const SpvReflectDescriptorBinding &spv_binding =
                    *spv_desc_set.bindings[k];

                auto &tmp_bindings = tmp_desc_sets[spv_desc_set.set];

                if (spv_binding.binding >= tmp_bindings.size()) {
                    tmp_bindings.resize(spv_binding.binding + 1,
                                        [](auto *ptr) {
                        ptr->type = BindingType::None;
                        ptr->stageUsage = 0;
                    });
                }
                    
                auto binding_type = 
                    convertSPVReflectBindingType(spv_binding.descriptor_type);

                auto &binding_out = tmp_bindings[spv_binding.binding];

                assert(binding_out.type == BindingType::None ||
                       binding_out.type == binding_type);

                binding_out.id = spv_binding.binding;
                binding_out.type = binding_type;
                binding_out.numResources = spv_binding.count;
                binding_out.stageUsage |= uint32_t(stage);
            }
        }
    }

    CountT total_num_bindings = 0;
    for (const auto &binding_arr : tmp_desc_sets) {
        total_num_bindings += binding_arr.size();
    }

    HeapArray<Binding> bindings_out(total_num_bindings);
    HeapArray<DescriptorSet> desc_sets_out(tmp_desc_sets.size());
    CountT cur_binding_offset = 0;

    for (CountT i = 0; i < desc_sets_out.size(); i++) {
        const auto &tmp_bindings = tmp_desc_sets[i];
        assert(tmp_bindings.size() != 0);

        desc_sets_out[i] = DescriptorSet {
            .bindingOffset = uint32_t(cur_binding_offset),
            .numBindings = uint32_t(tmp_bindings.size()),
        };

        memcpy(bindings_out.data() + cur_binding_offset,
               tmp_bindings.data(),
               tmp_bindings.size() * sizeof(Binding));

        cur_binding_offset += tmp_bindings.size();
    }

    spvReflectDestroyShaderModule(&rfl_mod);

    return SPIRV {
        .entryPoints = std::move(entry_points),
        .bindings = std::move(bindings_out),
        .descriptorSets = std::move(desc_sets_out),
    };
}

SPIRVShader ShaderCompiler::compileHLSLFileToSPV(
   const char *path,
   Span<const char *> include_dirs,
   Span<const char *> defines)
{
    CComPtr<IDxcBlobEncoding> src_blob =
        loadFileToDxcBlob(impl_->dxcUtils, path);

    auto spv_bytecode = hlslToSPV(src_blob->GetBufferPointer(),
        src_blob->GetBufferSize(),
        impl_->dxcCompiler, include_dirs, defines);

    refl::SPIRV refl = buildSPIRVReflectionData(spv_bytecode.data(),
        spv_bytecode.size() * sizeof(uint32_t));

    return SPIRVShader {
        .bytecode = std::move(spv_bytecode),
    };
}

}
