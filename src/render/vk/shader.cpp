#include "shader.hpp"
#include "utils.hpp"

#include <ShaderLang.h>
#include <GlslangToSpv.h>
#include <DirStackFileIncluder.h>
#include <ResourceLimits.h>

#include <spirv_reflect.h>

#include <algorithm>
#include <iostream>
#include <fstream>

#include <madrona/heap_array.hpp>

using namespace std;

namespace madrona {
namespace render {
namespace vk {

static void inline checkReflect(SpvReflectResult res, const char *msg)
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

    FATAL("%s: %s\n", msg, err_str);
}

#define REQ_RFL(expr) checkReflect((expr), LOC_APPEND(#expr))

static VkShaderStageFlagBits getStage(string_view name)
{
    string_view suffix = name.substr(name.rfind('.') + 1);

    if (suffix == "vert") {
        return VK_SHADER_STAGE_VERTEX_BIT;
    } else if (suffix == "frag") {
        return VK_SHADER_STAGE_FRAGMENT_BIT;
    } else if (suffix == "comp") {
        return VK_SHADER_STAGE_COMPUTE_BIT;
    } else {
        FATAL("Invalid shader stage");
    }
}

static vector<uint32_t> compileToSPV(const HeapArray<char> &src,
                                     VkShaderStageFlagBits vk_stage,
                                     const string &name,
                                     const string &shader_dir,
                                     const string &full_path,
                                     Span<const string> defines)
{
    EShLanguage stage;
    switch (vk_stage) {
        case VK_SHADER_STAGE_VERTEX_BIT: {
            stage = EShLangVertex;
        } break;
        case VK_SHADER_STAGE_FRAGMENT_BIT: {
            stage = EShLangFragment;
        } break;
        case VK_SHADER_STAGE_COMPUTE_BIT: {
            stage = EShLangCompute;
        } break;
        default: {
            FATAL("Unknown mapping from vulkan stage to glslang");
        }
    }

    glslang::TShader shader(stage);

    const char *src_ptr = src.data();
    int num_src_bytes = src.size();

    const char *debug_path = full_path.c_str();

    shader.setStringsWithLengthsAndNames(&src_ptr, &num_src_bytes,
                                         &debug_path, 1);

    string preamble = "";
    for (const string &def : defines) {
        preamble += "#define " + def + "\n";
    }
    shader.setPreamble(preamble.c_str());

    int vk_semantic_version = 100;
    glslang::EshTargetClientVersion vk_client_version =
        glslang::EShTargetVulkan_1_2;
    glslang::EShTargetLanguageVersion spv_version = glslang::EShTargetSpv_1_5;

    shader.setEnvInput(glslang::EShSourceGlsl, stage, glslang::EShClientVulkan,
                       vk_semantic_version);
    shader.setEnvClient(glslang::EShClientVulkan, vk_client_version);
    shader.setEnvTarget(glslang::EShTargetSpv, spv_version);

    // EshMsgDebugInfo is necessary in order to output
    // the main source file OpSource for some reason??
    EShMessages desired_msgs =
        (EShMessages)(EShMsgSpvRules | EShMsgVulkanRules | EShMsgDebugInfo);

    const TBuiltInResource *resource_limits = GetDefaultResources();

    DirStackFileIncluder preprocess_includer;
    preprocess_includer.pushExternalLocalDirectory(shader_dir);

    auto handleError = [&](const char *prefix) {
        FATAL("%s for shader: %s\n%s\n%s\n", prefix, name.c_str(),
              shader.getInfoLog(),
              shader.getInfoDebugLog());
    };

    if (!shader.parse(resource_limits, 110, false, desired_msgs,
                      preprocess_includer)) {
        handleError("Parsing failed");
    }

    glslang::TProgram prog;
    prog.addShader(&shader);

    if (!prog.link(desired_msgs)) {
        handleError("Linking failed");
    }

    vector<uint32_t> spv;
    spv::SpvBuildLogger spv_log;
    glslang::SpvOptions spv_opts;
    spv_opts.generateDebugInfo = true;

    glslang::GlslangToSpv(*prog.getIntermediate(stage), spv, &spv_log,
                          &spv_opts);

    return spv;
}

struct ReflectedSetInfo {
    struct BindingInfo {
        uint32_t id;
        VkDescriptorType type;
        uint32_t numDescriptors;
        VkShaderStageFlags stageUsage;
    };

    uint32_t id;
    vector<BindingInfo> bindings;
    uint32_t maxBindingID;
};

static vector<ReflectedSetInfo> getReflectionInfo(const vector<uint32_t> &spv,
                                                  VkShaderStageFlagBits stage)
{
    SpvReflectShaderModule rfl_mod;
    REQ_RFL(spvReflectCreateShaderModule(spv.size() * sizeof(uint32_t),
                                         spv.data(), &rfl_mod));

    uint32_t num_sets = 0;
    REQ_RFL(spvReflectEnumerateDescriptorSets(&rfl_mod, &num_sets, nullptr));

    if (num_sets == 0) {
        return {};
    }

    HeapArray<SpvReflectDescriptorSet *> desc_sets(num_sets);
    REQ_RFL(spvReflectEnumerateDescriptorSets(&rfl_mod, &num_sets,
                                              desc_sets.data()));

    vector<ReflectedSetInfo> sets;

    for (int set_idx = 0; set_idx < (int)num_sets; set_idx++) {
        SpvReflectDescriptorSet &rfl_set = *(desc_sets[set_idx]);
        ReflectedSetInfo set_info;
        set_info.id = rfl_set.set;
        set_info.bindings.reserve(rfl_set.binding_count);

        set_info.maxBindingID = 0;

        for (int binding_idx = 0; binding_idx < (int)rfl_set.binding_count;
             binding_idx++) {
            const SpvReflectDescriptorBinding &rfl_binding =
                *(rfl_set.bindings[binding_idx]);

            uint32_t num_descriptors = 1;
            for (int dim_idx = 0; dim_idx < (int)rfl_binding.array.dims_count;
                 dim_idx++) {
                num_descriptors *= rfl_binding.array.dims[dim_idx];
            }

            if (rfl_binding.binding > set_info.maxBindingID) {
                set_info.maxBindingID = rfl_binding.binding;
            }

            set_info.bindings.push_back({
                rfl_binding.binding,
                (VkDescriptorType)rfl_binding.descriptor_type,
                num_descriptors,
                stage,
            });
        }

        sets.emplace_back(move(set_info));
    }

    spvReflectDestroyShaderModule(&rfl_mod);

    return sets;
}

static void mergeReflectedSet(ReflectedSetInfo &dst,
                              const ReflectedSetInfo &src)
{
    for (const auto &new_binding : src.bindings) {
        bool match_found = false;
        for (auto &existing_binding : dst.bindings) {
            if (new_binding.id == existing_binding.id) {
                match_found = true;
                existing_binding.stageUsage |= new_binding.stageUsage;
                if (existing_binding.type != new_binding.type ||
                    existing_binding.numDescriptors !=
                        new_binding.numDescriptors) {
                    cerr << "Mismatched binding " << existing_binding.id
                         << " in set " << dst.id << endl;
                }
            }
        }

        if (!match_found) {
            dst.bindings.push_back(new_binding);
        }
    }
}

PipelineShaders::PipelineShaders(
    const DeviceState &d,
    Span<const string> shader_names,
    Span<const BindingOverride> binding_overrides,
    Span<const string> defines,
    const char *shader_dir)
    : dev(d),
      shaders_(),
      layouts_(),
      base_pool_sizes_()
{
    vector<ReflectedSetInfo> reflected_sets;

    for (const auto &shader_name : shader_names) {
        const string full_path = string(shader_dir) + shader_name;

        ifstream shader_file(full_path, ios::binary | ios::ate);

        streampos fend = shader_file.tellg();
        shader_file.seekg(0, ios::beg);
        streampos fbegin = shader_file.tellg();
        size_t file_size = fend - fbegin;

        if (file_size == 0) {
            FATAL("Empty shader file at %s", full_path.c_str());
        }

        HeapArray<char> shader_src(file_size);
        shader_file.read(shader_src.data(), file_size);

        VkShaderStageFlagBits stage = getStage(shader_name);

        vector<uint32_t> spv =
            compileToSPV(shader_src, stage, shader_name, shader_dir,
                         full_path, defines);

        VkShaderModuleCreateInfo shader_info;
        shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shader_info.pNext = nullptr;
        shader_info.flags = 0;
        shader_info.codeSize = spv.size() * sizeof(uint32_t);
        shader_info.pCode = spv.data();

        VkShaderModule shader_module;
        REQ_VK(dev.dt.createShaderModule(dev.hdl, &shader_info, nullptr,
                                         &shader_module));

        shaders_.push_back(shader_module);

        vector<ReflectedSetInfo> shader_sets = getReflectionInfo(spv, stage);

        for (const ReflectedSetInfo &shader_set : shader_sets) {
            bool match_found = false;
            for (ReflectedSetInfo &prior_set : reflected_sets) {
                if (prior_set.id == shader_set.id) {
                    match_found = true;
                    mergeReflectedSet(prior_set, shader_set);
                    break;
                }
            }
            if (!match_found) {
                reflected_sets.push_back(shader_set);
            }
        }
    }

    uint32_t max_set_id = 0;
    for (const auto &desc_set : reflected_sets) {
        if (desc_set.id > max_set_id) {
            max_set_id = desc_set.id;
        }
    }

    layouts_.resize(max_set_id + 1);
    base_pool_sizes_.resize(max_set_id + 1);

    vector<HeapArray<VkDescriptorSetLayoutBinding>> binding_infos;
    binding_infos.reserve(reflected_sets.size());

    vector<vector<VkDescriptorBindingFlags>> binding_flags;
    binding_flags.reserve(reflected_sets.size());

    for (const auto &desc_set : reflected_sets) {
        HeapArray<VkDescriptorSetLayoutBinding> set_binding_info(
            desc_set.bindings.size());

        for (int binding_idx = 0; binding_idx < (int)set_binding_info.size();
             binding_idx++) {
            const auto &rfl_binding = desc_set.bindings[binding_idx];
            auto &binding_info = set_binding_info[binding_idx];

            binding_info.binding = rfl_binding.id;
            binding_info.descriptorType = rfl_binding.type;
            binding_info.descriptorCount = rfl_binding.numDescriptors;
            binding_info.stageFlags = rfl_binding.stageUsage;
            binding_info.pImmutableSamplers = nullptr;
        }

        binding_infos.emplace_back(move(set_binding_info));
        binding_flags.emplace_back(binding_infos.back().size());
    }

    for (const auto &binding_override : binding_overrides) {
        if (binding_override.setID >= binding_infos.size()) {
            continue;
        }

        auto &set_bindings = binding_infos[binding_override.setID];

        if (binding_override.bindingID >= set_bindings.size()) {
            continue;
        }

        VkDescriptorSetLayoutBinding &binding =
            set_bindings[binding_override.bindingID];

        if (binding_override.sampler != VK_NULL_HANDLE) {
            binding.pImmutableSamplers = &binding_override.sampler;
        }

        binding.descriptorCount = binding_override.descriptorCount;

        binding_flags[binding_override.setID][binding_override.bindingID] =
            binding_override.flags;
    }

    for (int set_id = 0; set_id < (int)binding_infos.size(); set_id++) {
        const HeapArray<VkDescriptorSetLayoutBinding> &set_binding_info =
            binding_infos[set_id];

        VkDescriptorSetLayoutBindingFlagsCreateInfo flag_info;
        flag_info.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
        flag_info.pNext = nullptr;
        flag_info.bindingCount = set_binding_info.size();
        flag_info.pBindingFlags = binding_flags[set_id].data();

        VkDescriptorSetLayoutCreateInfo layout_info;
        layout_info.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.pNext = &flag_info;
        layout_info.flags = 0;
        layout_info.bindingCount = set_binding_info.size();
        layout_info.pBindings = set_binding_info.data();

        VkDescriptorSetLayout layout;
        REQ_VK(dev.dt.createDescriptorSetLayout(dev.hdl, &layout_info, nullptr,
                                                &layout));
        layouts_[set_id] = layout;

        auto &set_pool_sizes = base_pool_sizes_[set_id];
        for (const auto &binding : set_binding_info) {
            set_pool_sizes.push_back({
                binding.descriptorType,
                binding.descriptorCount,
            });
        }
    }
}

PipelineShaders::~PipelineShaders()
{
    for (VkShaderModule mod : shaders_) {
        dev.dt.destroyShaderModule(dev.hdl, mod, nullptr);
    }

    for (VkDescriptorSetLayout layout : layouts_) {
        dev.dt.destroyDescriptorSetLayout(dev.hdl, layout, nullptr);
    }
}

void PipelineShaders::initCompiler()
{
    glslang::InitializeProcess();
}

VkDescriptorPool PipelineShaders::makePool(uint32_t set_id,
                                          uint32_t max_sets) const
{
    const vector<VkDescriptorPoolSize> &base_sizes = base_pool_sizes_[set_id];

    vector<VkDescriptorPoolSize> pool_sizes;
    pool_sizes.reserve(base_sizes.size());

    for (const auto &base_size : base_sizes) {
        pool_sizes.push_back({
            base_size.type,
            base_size.descriptorCount * max_sets,
        });
    }

    VkDescriptorPoolCreateInfo pool_info;
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.pNext = nullptr;
    pool_info.flags = 0;
    pool_info.maxSets = max_sets;
    pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
    pool_info.pPoolSizes = pool_sizes.data();

    VkDescriptorPool pool;
    REQ_VK(dev.dt.createDescriptorPool(dev.hdl, &pool_info, nullptr, &pool));

    return pool;
}

}
}
}
