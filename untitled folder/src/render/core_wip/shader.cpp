#include "utils.hpp"

#include <algorithm>

#include <madrona/heap_array.hpp>

#include <madrona/render/vk/shader.hpp>

using namespace std;

namespace madrona::render::vk {

ParamBlockBuilder::ParamBlockBuilder()
    : type_counts_()
{
    utils::zeroN<int32_t>(type_counts_.data(), type_counts_.size());
}

void ParamBlockAllocator::addParamBlock(const ParamBlockDesc &desc,
                                        CountT num_blocks)
{
    for (CountT i = 0; i < desc.typeCounts.size(); i++) {
        type_counts_[i] += desc.typeCounts[i] * num_blocks;
    }
}

ParamBlockStore ParamBlockBuilder::build(ParamBlock *blocks_out)
{

}

Shader::Shader(Device &dev, StackAlloc &tmp_alloc,
               void *ir, CountT num_ir_bytes,
               const refl::ShaderInfo &refl_info)
    : Shader(dev, tmp_alloc, ir, num_ir_bytes,
             static_cast<const refl::SPIRV &>(refl_info))
{}

Shader::Shader(Device &dev, StackAlloc &tmp_alloc,
               void *ir, CountT num_ir_bytes,
               const refl::SPIRV &refl_info)
    : shader_(),
      set_layouts_(refl_info.descriptorSets.size()),
      set_type_counts_(refl_info.descriptorSets.size())
{
    VkShaderModuleCreateInfo shader_info;
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.pNext = nullptr;
    shader_info.flags = 0;
    shader_info.codeSize = num_ir_bytes;
    shader_info.pCode = (uint32_t *)ir;

    REQ_VK(dev.dt.createShaderModule(dev.hdl, &shader_info, nullptr,
                                     &shader_));

    for (CountT set_idx = 0; set_idx < refl_info.descriptorSets.size();
         set_idx++) {
        auto alloc_frame = tmp_alloc.push();

        const auto &desc_set = refl_info.descriptorSets[i];

        VkDescriptorSetLayoutBinding *set_binding_infos = tmp_alloc.allocN<
            VkDescriptorSetLayoutBinding>(desc_set.numBindings);

        VkDescriptorBindingFlags *set_binding_flags = tmp_alloc.allocN<
            VkDescriptorBindingFlags>(desc_set.numBindings);

        ParamBlockDesc::TypeCountArray type_counts;
        utils::zeroN<int32_t>(type_counts.data(), type_counts.size());

        for (CountT binding_idx = 0;
             binding_idx < (CountT)desc_set.numBindings; binding_idx++) {
            const auto &rfl_binding =
                refl_info.bindings[desc_set.bindingOffset + binding_idx];
            auto &binding_info = set_binding_info[binding_idx];

            if (rfl_binding.type == refl::BindingType::None) {
                continue;
            }

            type_counts[(uint32_t)rfl_binding.type - 1] += 1;

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

            set_binding_flags[binding_idx] = 0;
        }

        VkDescriptorSetLayoutBindingFlagsCreateInfo flag_info;
        flag_info.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
        flag_info.pNext = nullptr;
        flag_info.bindingCount = desc_set.numBindings;
        flag_info.pBindingFlags = set_binding_flags;

        VkDescriptorSetLayoutCreateInfo layout_info;
        layout_info.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.pNext = &flag_info;
        layout_info.flags = 0;
        layout_info.bindingCount = desc_set.numBindings;
        layout_info.pBindings = set_binding_infos;

        VkDescriptorSetLayout layout;
        REQ_VK(dev.dt.createDescriptorSetLayout(dev.hdl, &layout_info, nullptr,
                                                &layout));

        set_layouts_[set_id] = layout;
        set_type_counts_[set_id] = type_counts;

        tmp_alloc.pop(alloc_frame);
    }
}

void Shader::destroy(Device &dev)
{
    dev.dt.destroyShaderModule(dev.hdl, shader_, nullptr);

    for (ParamBlockDesc &desc : set_descs_) {
        dev.dt.destroyDescriptorSetLayout(dev.hdl, desc.layout, nullptr);
    }
}

}
