#include "utils.hpp"

#include <algorithm>

#include <madrona/heap_array.hpp>

using namespace std;

namespace madrona::render::vk {

Shader::Shader(Device &dev,
               void *ir, CountT num_ir_bytes,
               const refl::ShaderInfo &reflection_base)
    : Shader(dev, ir, num_ir_bytes,
             static_cast<const refl::SPIRV &>(reflection_base))
{}

Shader::Shader(Device &dev,
               void *ir, CountT num_ir_bytes,
               const refl::SPIRV &refl_info)
    : dev(d),
      layouts_(refl_info.descriptorSets.size()),
      base_pool_sizes_(refl_info.descriptorSets.size())
{
    VkShaderModuleCreateInfo shader_info;
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.pNext = nullptr;
    shader_info.flags = 0;
    shader_info.codeSize = num_ir_bytes;
    shader_info.pCode = (uint32_t *)ir;

    REQ_VK(dev.dt.createShaderModule(dev.hdl, &shader_info, nullptr,
                                     &shader_));

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

            set_binding_info.push_back(binding_info);
        }

        binding_infos.emplace_back(std::move(set_binding_info));
        binding_flags.emplace_back(binding_infos.back().size());
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

void Shader::destroy(Device &dev)
{
    dev.dt.destroyShaderModule(dev.hdl, shader_, nullptr);

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
