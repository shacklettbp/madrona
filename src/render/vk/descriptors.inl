#pragma once

namespace madrona {
namespace render {
namespace vk {

void DescHelper::update(const Device &dev,
                        const VkWriteDescriptorSet *updates,
                        uint32_t num_desc_updates)
{
    dev.dt.updateDescriptorSets(dev.hdl, num_desc_updates, updates, 0,
                                nullptr);
}

void DescHelper::textures(VkWriteDescriptorSet &update,
                          VkDescriptorSet desc_set,
                          VkDescriptorImageInfo *imgs,
                          uint32_t num_textures,
                          uint32_t binding,
                          uint32_t arr_offset)
{
    update.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    update.pNext = nullptr;
    update.dstSet = desc_set;
    update.dstBinding = binding;
    update.dstArrayElement = arr_offset;
    update.descriptorCount = num_textures;
    update.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    update.pImageInfo = imgs;
    update.pBufferInfo = nullptr;
    update.pTexelBufferView = nullptr;
}

void DescHelper::buffer(VkWriteDescriptorSet &update,
                        VkDescriptorSet desc_set,
                        const VkDescriptorBufferInfo *buf,
                        uint32_t binding,
                        VkDescriptorType type)
{
    update.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    update.pNext = nullptr;
    update.dstSet = desc_set;
    update.dstBinding = binding;
    update.dstArrayElement = 0;
    update.descriptorCount = 1;
    update.descriptorType = type;
    update.pImageInfo = nullptr;
    update.pBufferInfo = buf;
    update.pTexelBufferView = nullptr;
}

void DescHelper::uniform(VkWriteDescriptorSet &update,
                         VkDescriptorSet desc_set,
                         const VkDescriptorBufferInfo *buf,
                         uint32_t binding)
{
    buffer(update, desc_set, buf, binding, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
}

void DescHelper::storage(VkWriteDescriptorSet &update,
                         VkDescriptorSet desc_set,
                         const VkDescriptorBufferInfo *buf,
                         uint32_t binding)
{
    buffer(update, desc_set, buf, binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
}

void DescHelper::storageImage(VkWriteDescriptorSet &update,
                              VkDescriptorSet desc_set,
                              const VkDescriptorImageInfo *img,
                              uint32_t binding,
                              uint32_t arr_offset)
{
    update.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    update.pNext = nullptr;
    update.dstSet = desc_set;
    update.dstBinding = binding;
    update.dstArrayElement = arr_offset;
    update.descriptorCount = 1;
    update.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    update.pImageInfo = img;
    update.pBufferInfo = nullptr;
    update.pTexelBufferView = nullptr;
}

void DescHelper::accelStructs(
    VkWriteDescriptorSet &update,
    VkWriteDescriptorSetAccelerationStructureKHR &as_update,
    VkDescriptorSet desc_set,
    const VkAccelerationStructureKHR *accel_structs,
    uint32_t num_accel_structs,
    uint32_t binding,
    uint32_t arr_offset)
{
    as_update.sType =
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    as_update.pNext = nullptr;
    as_update.accelerationStructureCount = num_accel_structs;
    as_update.pAccelerationStructures = accel_structs;

    update.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    update.pNext = &as_update;
    update.dstSet = desc_set;
    update.dstBinding = binding;
    update.dstArrayElement = arr_offset;
    update.descriptorCount = num_accel_structs;
    update.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    update.pImageInfo = nullptr;
    update.pBufferInfo = nullptr;
    update.pTexelBufferView = nullptr;
}

}
}
}
