#pragma once

namespace madrona {
namespace render {
namespace vk {

DescriptorUpdates::DescriptorUpdates(uint32_t max_updates) : updates_()
{
    updates_.reserve(max_updates);
}

void DescriptorUpdates::update(const DeviceState &dev)
{
    dev.dt.updateDescriptorSets(dev.hdl, updates_.size(), updates_.data(), 0,
                                nullptr);
}

void DescriptorUpdates::textures(VkDescriptorSet desc_set,
                                 VkDescriptorImageInfo *imgs,
                                 uint32_t num_textures,
                                 uint32_t binding,
                                 uint32_t arr_elem)
{
    VkWriteDescriptorSet desc_update;
    desc_update.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    desc_update.pNext = nullptr;
    desc_update.dstSet = desc_set;
    desc_update.dstBinding = binding;
    desc_update.dstArrayElement = arr_elem;
    desc_update.descriptorCount = num_textures;
    desc_update.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    desc_update.pImageInfo = imgs;
    desc_update.pBufferInfo = nullptr;
    desc_update.pTexelBufferView = nullptr;

    updates_.push_back(desc_update);
}

void DescriptorUpdates::buffer(VkDescriptorSet desc_set,
                               const VkDescriptorBufferInfo *buf,
                               uint32_t binding,
                               VkDescriptorType type)
{
    VkWriteDescriptorSet desc_update;
    desc_update.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    desc_update.pNext = nullptr;
    desc_update.dstSet = desc_set;
    desc_update.dstBinding = binding;
    desc_update.dstArrayElement = 0;
    desc_update.descriptorCount = 1;
    desc_update.descriptorType = type;
    desc_update.pImageInfo = nullptr;
    desc_update.pBufferInfo = buf;
    desc_update.pTexelBufferView = nullptr;

    updates_.push_back(desc_update);
}

void DescriptorUpdates::uniform(VkDescriptorSet desc_set,
                                const VkDescriptorBufferInfo *buf,
                                uint32_t binding)
{
    buffer(desc_set, buf, binding, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
}

void DescriptorUpdates::storage(VkDescriptorSet desc_set,
                                const VkDescriptorBufferInfo *buf,
                                uint32_t binding)
{
    buffer(desc_set, buf, binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
}

}
}
}
