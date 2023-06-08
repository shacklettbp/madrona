#include "shader.hpp"
#include "utils.hpp"

#include <algorithm>
#include <iostream>
#include <fstream>

#include <madrona/heap_array.hpp>

using namespace std;

namespace madrona::render::vk {

PipelineShaders::PipelineShaders(
        const DeviceState &d,
        const SPIRVShader &shader,
        Span<const BindingOverride> binding_overrides)
    : dev(d),
      shaders_(),
      layouts_(),
      base_pool_sizes_()
{
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

    const refl::SPIRV &refl_info = shader.reflectionInfo;

    layouts_.resize(refl_info.descriptorSets.size());
    base_pool_sizes_.resize(refl_info.descriptorSets.size());

    vector<vector<VkDescriptorSetLayoutBinding>> binding_infos;
    binding_infos.reserve(refl_info.descriptorSets.size());

    vector<vector<VkDescriptorBindingFlags>> binding_flags;
    binding_flags.reserve(refl_info.descriptorSets.size());

    for (const auto &desc_set : refl_info.descriptorSets) {
        vector<VkDescriptorSetLayoutBinding> set_binding_info;
        set_binding_info.reserve(desc_set.numBindings);

        for (CountT binding_idx = 0;
             binding_idx < (CountT)desc_set.numBindings; binding_idx++) {
            const auto &rfl_binding =
                refl_info.bindings[desc_set.bindingOffset + binding_idx];
            auto &binding_info = set_binding_info[binding_idx];

            if (rfl_binding.type == refl::BindingType::None) {
                continue;
            }

            binding_info.binding = rfl_binding.id;

            VkDescriptorType desc_type;
            switch (rfl_binding.type) {
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
                case refl::BindingType::AccelerationStructure: {
                    desc_type =
                        VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
                } break;
                default: {
                    FATAL("Unsupported binding type");
                } break;
            }

            binding_info.descriptorType = desc_type;
            binding_info.descriptorCount = rfl_binding.numResources;
            binding_info.stageFlags = 0;
            if ((rfl_binding.stageUsage & uint32_t(refl::Stage::Vertex))) {
                binding_info.stageFlags |= VK_SHADER_STAGE_VERTEX_BIT;
            }
            if ((rfl_binding.stageUsage & uint32_t(refl::Stage::Fragment))) {
                binding_info.stageFlags |= VK_SHADER_STAGE_FRAGMENT_BIT;
            }
            if ((rfl_binding.stageUsage & uint32_t(refl::Stage::Compute))) {
                binding_info.stageFlags |= VK_SHADER_STAGE_COMPUTE_BIT;
            }

            binding_info.pImmutableSamplers = nullptr;
        }

        binding_infos.emplace_back(std::move(set_binding_info));
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
        const auto &set_binding_info = binding_infos[set_id];

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
