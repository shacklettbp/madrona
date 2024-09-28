#include "pipeline_shaders.hpp"
#include "utils.hpp"

#include <algorithm>
#include <iostream>
#include <fstream>

#include <madrona/heap_array.hpp>

using namespace std;

namespace madrona::render::vk {

static VkDescriptorType bindingTypeToDescriptorType(
    refl::BindingType binding_type)
{
    VkDescriptorType desc_type;
    switch (binding_type) {
        case refl::BindingType::Sampler: {
            desc_type = VK_DESCRIPTOR_TYPE_SAMPLER;
        } break;
        case refl::BindingType::Texture: {
            desc_type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        } break;
        case refl::BindingType::UniformBuffer: {
            desc_type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        } break;
        case refl::BindingType::StorageBuffer: {
            desc_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        } break;
        case refl::BindingType::StorageImage: {
            desc_type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        } break;
        case refl::BindingType::AccelerationStructure: {
            desc_type =
                VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        } break;
        default: {
            FATAL("Unsupported binding type");
        } break;
    }

    return desc_type;
}

static VkShaderStageFlags buildStageUsageFlags(uint32_t rfl_stage_usage)
{
    VkShaderStageFlags flags = 0;
    if ((rfl_stage_usage & uint32_t(ShaderStage::Vertex))) {
        flags |= VK_SHADER_STAGE_VERTEX_BIT;
    }
    if ((rfl_stage_usage & uint32_t(ShaderStage::Fragment))) {
        flags |= VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    if ((rfl_stage_usage & uint32_t(ShaderStage::Compute))) {
        flags |= VK_SHADER_STAGE_COMPUTE_BIT;
    }

    return flags;
}

PipelineShaders::PipelineShaders(
        const Device &d,
        StackAlloc &tmp_alloc,
        Span<const SPIRVShader> shaders,
        Span<const BindingOverride> binding_overrides,
        VkShaderStageFlags extra_stages)
    : dev(d),
      shaders_(),
      layouts_(),
      base_pool_sizes_()
{
    for (const SPIRVShader &shader : shaders) {
        VkShaderModuleCreateInfo shader_info;
        shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shader_info.pNext = nullptr;
        shader_info.flags = 0;
        shader_info.codeSize = shader.bytecode.size() * sizeof(uint32_t);
        shader_info.pCode = shader.bytecode.data();

        VkShaderModule shader_module;
        REQ_VK(dev.dt.createShaderModule(dev.hdl, &shader_info, nullptr,
                                         &shader_module));

        shaders_.push_back(shader_module);
    }

    AllocFrame alloc_frame = tmp_alloc.push();

    CountT merged_set_arr_size = 0;
    for (const SPIRVShader &shader : shaders) {
        const refl::SPIRV &rfl_info = shader.reflectionInfo;

        for (const auto &desc_set : rfl_info.descriptorSets) {
            merged_set_arr_size =
                std::max((CountT)desc_set.id + 1, merged_set_arr_size);
        }
    }

    uint32_t *set_binding_arr_sizes =
        tmp_alloc.allocN<uint32_t>(merged_set_arr_size);
    utils::zeroN<uint32_t>(set_binding_arr_sizes, merged_set_arr_size);

    for (const SPIRVShader &shader : shaders) {
        const refl::SPIRV &rfl_info = shader.reflectionInfo;

        for (const auto &desc_set : rfl_info.descriptorSets) {
            for (CountT binding_idx = 0;
                 binding_idx < (CountT)desc_set.numBindings; binding_idx++) {
                const auto &rfl_binding =
                    rfl_info.bindings[desc_set.bindingOffset + binding_idx];

                if (rfl_binding.type == refl::BindingType::None) {
                    continue;
                }

                set_binding_arr_sizes[desc_set.id] = std::max(
                    rfl_binding.id + 1, set_binding_arr_sizes[desc_set.id]);
            }
        }
    }

    VkDescriptorSetLayoutBinding **set_binding_infos =
        tmp_alloc.allocN<VkDescriptorSetLayoutBinding *>(merged_set_arr_size);

    VkDescriptorBindingFlags **set_binding_flags =
        tmp_alloc.allocN<VkDescriptorBindingFlags *>(merged_set_arr_size);

    constexpr uint32_t binding_sentinel = 0xFFFF'FFFF_u32;

    for (CountT set_id = 0; set_id < merged_set_arr_size; set_id++) {
        uint32_t set_binding_arr_size = set_binding_arr_sizes[set_id];

        if (set_binding_arr_size == 0) {
            continue;
        }

        set_binding_infos[set_id] =
            tmp_alloc.allocN<VkDescriptorSetLayoutBinding>(
                set_binding_arr_size);

        for (uint32_t binding_idx = 0; binding_idx < set_binding_arr_size;
             binding_idx++) {
            set_binding_infos[set_id][binding_idx].binding = binding_sentinel;
        }

        set_binding_flags[set_id] = tmp_alloc.allocN<VkDescriptorBindingFlags>(
            set_binding_arr_size);
        utils::zeroN<VkDescriptorBindingFlags>(
            set_binding_flags[set_id], set_binding_arr_size);
    }

    for (const SPIRVShader &shader : shaders) {
        const refl::SPIRV &rfl_info = shader.reflectionInfo;

        for (const auto &desc_set : rfl_info.descriptorSets) {
            for (CountT binding_idx = 0;
                 binding_idx < (CountT)desc_set.numBindings; binding_idx++) {
                const auto &rfl_binding =
                    rfl_info.bindings[desc_set.bindingOffset + binding_idx];

                if (rfl_binding.type == refl::BindingType::None) {
                    continue;
                }

                VkShaderStageFlags binding_stage_flags =
                    buildStageUsageFlags(rfl_binding.stageUsage);

                VkDescriptorType binding_desc_type =
                    bindingTypeToDescriptorType(rfl_binding.type);

                VkDescriptorSetLayoutBinding &binding_info =
                    set_binding_infos[desc_set.id][rfl_binding.id];

                if (binding_info.binding != binding_sentinel) {
                    if (binding_info.descriptorType != binding_desc_type ||
                        binding_info.descriptorCount !=
                            rfl_binding.numResources) {
                        FATAL("Descriptor mismatch between stages");
                    }
                    binding_info.stageFlags |= binding_stage_flags;
                    
                    continue;
                }

                binding_info.binding = rfl_binding.id;
                binding_info.descriptorType = binding_desc_type;
                binding_info.descriptorCount = rfl_binding.numResources;
                binding_info.stageFlags = binding_stage_flags | extra_stages;
                binding_info.pImmutableSamplers = nullptr;
            }
        }
    }

    for (const auto &binding_override : binding_overrides) {
        uint32_t set_id = binding_override.setID;
        uint32_t binding_id = binding_override.bindingID;

        if (set_id >= merged_set_arr_size) {
            FATAL("Invalid binding override set ID");
        }
        if (binding_id >= set_binding_arr_sizes[set_id]) {
            FATAL("Invalid binding override binding ID");
        }

        VkDescriptorSetLayoutBinding &binding =
            set_binding_infos[set_id][binding_id];

        if (binding_override.sampler != VK_NULL_HANDLE) {
            binding.pImmutableSamplers = &binding_override.sampler;
        }

        binding.descriptorCount = binding_override.descriptorCount;

        set_binding_flags[set_id][binding_id] = binding_override.flags;
    }

    layouts_.resize(merged_set_arr_size);
    base_pool_sizes_.resize(merged_set_arr_size);

    for (CountT set_id = 0; set_id < merged_set_arr_size; set_id++) {
        const auto *set_bindings = set_binding_infos[set_id];
        if (set_bindings == nullptr) {
            layouts_[set_id] = VK_NULL_HANDLE;
            continue;
        }

        CountT set_binding_arr_size = set_binding_arr_sizes[set_id];

        // Compact the binding info and flag arrays
        uint32_t num_active_bindings = 0;
        for (CountT i = 0; i < set_binding_arr_size; i++) {
            const VkDescriptorSetLayoutBinding &binding_info =
                set_binding_infos[set_id][i];

            if (binding_info.binding == binding_sentinel) {
                continue;
            }

            if (i != num_active_bindings) {
                set_binding_infos[set_id][num_active_bindings] = binding_info;
                set_binding_flags[set_id][num_active_bindings] =
                    set_binding_flags[set_id][i];
            }

            num_active_bindings += 1;
        }

        VkDescriptorSetLayoutBindingFlagsCreateInfo flag_info;
        flag_info.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
        flag_info.pNext = nullptr;
        flag_info.bindingCount = num_active_bindings;
        flag_info.pBindingFlags = set_binding_flags[set_id];

        VkDescriptorSetLayoutCreateInfo layout_info;
        layout_info.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.pNext = &flag_info;
        layout_info.flags = 0;
        layout_info.bindingCount = num_active_bindings;
        layout_info.pBindings = set_binding_infos[set_id];

        VkDescriptorSetLayout layout;
        REQ_VK(dev.dt.createDescriptorSetLayout(dev.hdl, &layout_info, nullptr,
                                                &layout));
        layouts_[set_id] = layout;

        // FIXME: better pool solution
        auto &set_pool_sizes = base_pool_sizes_[set_id];
        for (CountT i = 0; i < (CountT)num_active_bindings; i++) {
            const auto &binding = set_binding_infos[set_id][i];

            set_pool_sizes.push_back({
                binding.descriptorType,
                binding.descriptorCount,
            });
        }
    }

    tmp_alloc.pop(alloc_frame);
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
