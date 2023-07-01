#include <madrona/render/shader_compiler.hpp>

#include <madrona/macros.hpp>
#include <madrona/optional.hpp>
#include <madrona/heap_array.hpp>
#include <madrona/dyn_array.hpp>

#include <spirv_reflect.h>

#include <cstdlib>
#include <memory>
#include <codecvt>

#if defined(MADRONA_WINDOWS)
// Cannot include dxc/dxcapi.h without first including windows.h
#include <windows.h>

// DXC doesn't define this on windows for some reason
#define DXC_FAILED(hr) FAILED(hr)
#endif

// On clang on linux and mac need to either compile with -fms-exceptions
// define __EMULATE_UUID. The latter seems simpler and matches GCC
#if defined(MADRONA_LINUX) or defined(MADRONA_APPLE)
#define __EMULATE_UUID 1
#endif
#include <dxc/dxcapi.h>
#undef __EMULATE_UUID

namespace madrona::render {

// DXC is broken on exit (hangs). Hack to stop the shared lib from unloading
// at program termination.
// https://github.com/microsoft/DirectXShaderCompiler/issues/5119
#if defined(MADRONA_LINUX) or defined(MADRONA_APPLE)
static __attribute__((constructor)) void dxcNoFreeHack()
{
    void *lib = dlopen("libdxcompiler."
#ifdef MADRONA_LINUX
                          "so"
#endif

#ifdef MADRONA_APPLE
                          "dylib"
#endif
                       , RTLD_NOW | RTLD_NOLOAD | RTLD_NODELETE);
    assert(lib != nullptr);
    dlclose(lib);
}
#endif

// Minimal class to avoid need to depend on ATL in windows
// Note that that so far we don't need to call AddRef because DXC returns the
// pointers with the reference count incremented.
template <typename T>
class COMUnique {
public:
    COMUnique()
        : p_(nullptr)
    {
        static_assert(std::is_standard_layout_v<COMUnique<T>>);
    }

    ~COMUnique()
    {
        if (p_ != nullptr) {
            static_cast<IUnknown *>(p_)->Release();
        }
    }

    COMUnique(const COMUnique<T> &) = delete;
    COMUnique(COMUnique &&o)
        : p_(o.p_)
    {
        o.p_ = nullptr;
    }

    T & operator*() { return *p_; }
    T * operator->() { return p_; }
    T ** operator&() { return &p_; }

    operator T *() { return p_; }

private:
    T *p_;
};

struct ShaderCompiler::Impl {
    COMUnique<IDxcUtils> dxcUtils;
    COMUnique<IDxcIncludeHandler> dxcIncludeHandler;
    COMUnique<IDxcCompiler3> dxcCompiler;
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
        COMUnique<IDxcUtils> dxc_utils;
        REQ_DXC(DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&dxc_utils)),
                "Failed to initialize DxcUtils");

        COMUnique<IDxcIncludeHandler> dxc_inc_handler;
        REQ_DXC(dxc_utils->CreateDefaultIncludeHandler(&dxc_inc_handler),
                "Failed to initialize DxcIncludeHandler");

        COMUnique<IDxcCompiler3> dxc_compiler;
        REQ_DXC(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&dxc_compiler)),
                "Failed to initialize DxcCompiler");

        return std::unique_ptr<ShaderCompiler::Impl>(new ShaderCompiler::Impl {
            std::move(dxc_utils),
            std::move(dxc_inc_handler),
            std::move(dxc_compiler),
        });
    }())
{}

ShaderCompiler::~ShaderCompiler() = default;

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

static COMUnique<IDxcBlobEncoding> loadFileToDxcBlob(
    IDxcUtils *dxc_utils,
    const char *shader_path)
{
    HeapArray<wchar_t> lshader_path = toWide(shader_path);

    uint32_t src_cp = CP_UTF8;
    COMUnique<IDxcBlobEncoding> blob;
    REQ_DXC(dxc_utils->LoadFile(lshader_path.data(), &src_cp, &blob),
            "Failed to load shader file");

    return blob;
}

static HeapArray<uint32_t> hlslToSPV(
    LPVOID src_shader_buffer,
    SIZE_T src_shader_len,
    const char *shader_path,
    IDxcCompiler3 *dxc_compiler,
    IDxcIncludeHandler *include_handler,
    Span<const char *const> include_dirs,
    Span<const ShaderCompiler::MacroDefn> macro_defns,
    const char *entry_func,
    ShaderStage entry_stage)
{
#define TYPE_STR(type_str) (type_str L"_6_7")
    LPCWSTR stage_type_str;
    if (entry_func == nullptr) {
        stage_type_str = TYPE_STR(L"lib");
    } else {
        switch (entry_stage) {
            case ShaderStage::Vertex:
                stage_type_str = TYPE_STR(L"vs");
                break;
            case ShaderStage::Fragment:
                stage_type_str = TYPE_STR(L"ps");
                break;
            case ShaderStage::Compute:
                stage_type_str = TYPE_STR(L"cs");
                break;
            case ShaderStage::Mesh:
                stage_type_str = TYPE_STR(L"ms");
                break;
            case ShaderStage::Amplification:
                stage_type_str = TYPE_STR(L"as");
                break;
            default:
                FATAL("Shader compilation: Unsupported shader stage type");
        }
    }
#undef TYPE_STR

    HeapArray<wchar_t> lshader_path = toWide(shader_path);

    DynArray<LPCWSTR> dxc_args {
        lshader_path.data(),
        DXC_ARG_WARNINGS_ARE_ERRORS,
        DXC_ARG_DEBUG,
        DXC_ARG_PACK_MATRIX_COLUMN_MAJOR,
        L"-spirv",
        L"-fspv-target-env=vulkan1.2",
        //L"-fspv-reflect",
        L"-T", stage_type_str,
        L"-HV", L"2021",
        L"-enable-16bit-types",
        L"-fspv-preserve-bindings"
    };

    Optional<HeapArray<wchar_t>> wentry = Optional<HeapArray<wchar_t>>::none();
    if (entry_func != nullptr) {
        wentry.emplace(toWide(entry_func));
        dxc_args.push_back(L"-E");
        dxc_args.push_back(wentry->data());
    }

    DynArray<HeapArray<wchar_t>> wdefines(macro_defns.size());

    for (const ShaderCompiler::MacroDefn &defn : macro_defns) {
        std::string combined;

        if (defn.value == nullptr) {
            combined = defn.name;
        } else {
            combined = std::string(defn.name) + "=" + defn.value;
        }

        wdefines.emplace_back(toWide(combined.c_str()));
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

    COMUnique<IDxcResult> compile_result;
    REQ_DXC(dxc_compiler->Compile(&src_info, dxc_args.data(),
                                  (uint32_t)dxc_args.size(),
                                  include_handler,
                                  IID_PPV_ARGS(&compile_result)),
            "Failed to compile shader");

    COMUnique<IDxcBlobUtf8> compile_errors;
    REQ_DXC(compile_result->GetOutput(
            DXC_OUT_ERRORS, IID_PPV_ARGS(&compile_errors), nullptr),
        "Failed to get DXC errors");

    if (compile_errors && compile_errors->GetStringLength() > 0) {
        FATAL("Compilation failed: %s", compile_errors->GetBufferPointer());
    }

    COMUnique<IDxcBlob> dxc_spv;
    REQ_DXC(compile_result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&dxc_spv),
            nullptr),
        "Failed to get SPIRV object from DXC");

    uint32_t num_shader_bytes = dxc_spv->GetBufferSize();
    assert(num_shader_bytes % 4 == 0);

    HeapArray<uint32_t> bytecode(num_shader_bytes / 4);
    memcpy(bytecode.data(), dxc_spv->GetBufferPointer(),
           num_shader_bytes);

    return bytecode;
}

static ShaderStage convertSPVReflectStage(
    SpvReflectShaderStageFlagBits stage)
{
    using namespace refl;

    switch (stage) {
        case SPV_REFLECT_SHADER_STAGE_VERTEX_BIT:
            return ShaderStage::Vertex;
        case SPV_REFLECT_SHADER_STAGE_FRAGMENT_BIT:
            return ShaderStage::Fragment;
        case SPV_REFLECT_SHADER_STAGE_COMPUTE_BIT:
            return ShaderStage::Compute;
        case SPV_REFLECT_SHADER_STAGE_TASK_BIT_EXT:
            return ShaderStage::Amplification;
        case SPV_REFLECT_SHADER_STAGE_MESH_BIT_EXT:
            return ShaderStage::Mesh;
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
        case SPV_REFLECT_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            return BindingType::StorageImage;
        case SPV_REFLECT_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
            return BindingType::AccelerationStructure;
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

        ShaderStage stage = convertSPVReflectStage(spv_rfl_entry.shader_stage);
        entry_points.emplace(i, EntryPoint {
            .name = spv_rfl_entry.name,
            .stage = stage,
        });

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

    CountT num_sets = 0;
    for (const auto &bindings : tmp_desc_sets) {
        if (bindings.size() > 0) {
            num_sets += 1;
        }
    }

    HeapArray<Binding> bindings_out(total_num_bindings);
    HeapArray<DescriptorSet> desc_sets_out(num_sets);

    CountT cur_binding_offset = 0;
    CountT cur_set_offset = 0;
    for (CountT i = 0; i < tmp_desc_sets.size(); i++) {
        const auto &tmp_bindings = tmp_desc_sets[i];

        // This descriptor set isn't actually used
        if (tmp_bindings.size() == 0) {
            continue;
        }

        desc_sets_out[cur_set_offset++] = DescriptorSet {
            .id = uint32_t(i),
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
        { std::move(entry_points) },
        std::move(bindings_out),
        std::move(desc_sets_out),
    };
}

SPIRVShader ShaderCompiler::compileHLSLFileToSPV(
   const char *path,
   Span<const char *const> include_dirs,
   Span<const MacroDefn> macro_defns,
   EntryConfig entry)
{
    COMUnique<IDxcBlobEncoding> src_blob =
        loadFileToDxcBlob(impl_->dxcUtils, path);

    auto spv_bytecode = hlslToSPV(src_blob->GetBufferPointer(),
        src_blob->GetBufferSize(), path, impl_->dxcCompiler,
        impl_->dxcIncludeHandler, include_dirs, macro_defns,
        entry.func, entry.stage);

    refl::SPIRV refl = buildSPIRVReflectionData(spv_bytecode.data(),
        spv_bytecode.size() * sizeof(uint32_t));

    return SPIRVShader {
        .bytecode = std::move(spv_bytecode),
        .reflectionInfo = std::move(refl),
    };
}

#ifdef MADRONA_APPLE
MTLShader ShaderCompiler::compileHLSLFileToMTL(
   const char *path,
   Span<const char *const> include_dirs,
   Span<const MacroDefn> macro_defns)
{
    COMUnique<IDxcBlobEncoding> src_blob =
        loadFileToDxcBlob(impl_->dxcUtils, path);

    (void)src_blob;
    (void)include_dirs;
    (void)macro_defns;

    return MTLShader {
        .bytecode = HeapArray<char>(0),
    };
}
#endif

}
