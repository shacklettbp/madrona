#pragma once

namespace madrona {
namespace render {
namespace vk {

QueueState::QueueState(VkQueue queue_hdl, bool shared)
    : queue_hdl_(queue_hdl),
      shared_(shared),
      mutex_()
{}

void QueueState::submit(const Device &dev,
                        uint32_t submit_count,
                        const VkSubmitInfo *pSubmits,
                        VkFence fence) const
{
    if (shared_) {
        mutex_.lock();
    }

    REQ_VK(dev.dt.queueSubmit(queue_hdl_, submit_count, pSubmits, fence));

    if (shared_) {
        mutex_.unlock();
    }
} 

void QueueState::bindSubmit(const Device &dev,
                            uint32_t submit_count,
                            const VkBindSparseInfo *pSubmits,
                            VkFence fence) const
{
    if (shared_) {
        mutex_.lock();
    }

    REQ_VK(dev.dt.queueBindSparse(queue_hdl_, submit_count, pSubmits, fence));

    if (shared_) {
        mutex_.unlock();
    }
}

bool QueueState::presentSubmit(const Device &dev,
                               const VkPresentInfoKHR *present_info) const
{
    if (shared_) {
        mutex_.lock();
    }

    // FIXME resize
    VkResult result = dev.dt.queuePresentKHR(queue_hdl_, present_info);

    if (shared_) {
        mutex_.unlock();
    }

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        return false;
    } else if (result == VK_SUBOPTIMAL_KHR) {
        return false;
    } else if (result != VK_SUCCESS) {
        assert(false);
    }

    return true;
}

VkDeviceAddress getDevAddr(const Device &dev, VkBuffer buf)
{
    VkBufferDeviceAddressInfoKHR addr_info;
    addr_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO_KHR;
    addr_info.pNext = nullptr;
    addr_info.buffer = buf;
    return dev.dt.getBufferDeviceAddress(dev.hdl, &addr_info);
}

VkCommandPool makeCmdPool(const Device &dev, uint32_t qf_idx)
{
    VkCommandPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = qf_idx;

    VkCommandPool pool;
    REQ_VK(dev.dt.createCommandPool(dev.hdl, &pool_info, nullptr, &pool));
    return pool;
}

VkCommandBuffer makeCmdBuffer(const Device &dev,
                              VkCommandPool pool,
                              VkCommandBufferLevel level)
{
    VkCommandBufferAllocateInfo info;
    info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    info.pNext = nullptr;
    info.commandPool = pool;
    info.level = level;
    info.commandBufferCount = 1;

    VkCommandBuffer cmd;
    REQ_VK(dev.dt.allocateCommandBuffers(dev.hdl, &info, &cmd));

    return cmd;
}

VkQueue makeQueue(const Device &dev, uint32_t qf_idx, uint32_t queue_idx)
{
    VkDeviceQueueInfo2 queue_info;
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2;
    queue_info.pNext = nullptr;
    queue_info.flags = 0;
    queue_info.queueFamilyIndex = qf_idx;
    queue_info.queueIndex = queue_idx;

    VkQueue queue;
    dev.dt.getDeviceQueue2(dev.hdl, &queue_info, &queue);

    return queue;
}

VkSemaphore makeBinarySemaphore(const Device &dev)
{
    VkSemaphoreCreateInfo sema_info;
    sema_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    sema_info.pNext = nullptr;
    sema_info.flags = 0;

    VkSemaphore sema;
    REQ_VK(dev.dt.createSemaphore(dev.hdl, &sema_info, nullptr, &sema));

    return sema;
}

VkSemaphore makeBinaryExternalSemaphore(const Device &dev)
{
    VkExportSemaphoreCreateInfo export_info;
    export_info.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
    export_info.pNext = nullptr;
    export_info.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkSemaphoreCreateInfo sema_info;
    sema_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    sema_info.pNext = &export_info;
    sema_info.flags = 0;

    VkSemaphore sema;
    REQ_VK(dev.dt.createSemaphore(dev.hdl, &sema_info, nullptr, &sema));

    return sema;
}

VkFence makeFence(const Device &dev, bool pre_signal)
{
    VkFenceCreateInfo fence_info;
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.pNext = nullptr;
    if (pre_signal) {
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    } else {
        fence_info.flags = 0;
    }

    VkFence fence;
    REQ_VK(dev.dt.createFence(dev.hdl, &fence_info, nullptr, &fence));

    return fence;
}

void waitForFenceInfinitely(const Device &dev, VkFence fence)
{
    VkResult res;
    while ((res = dev.dt.waitForFences(dev.hdl, 1, &fence, VK_TRUE, ~0ull)) !=
           VK_SUCCESS) {
        if (res != VK_TIMEOUT) {
            REQ_VK(res);
        }
    }
}

void resetFence(const Device &dev, VkFence fence)
{
    dev.dt.resetFences(dev.hdl, 1, &fence);
}

VkDescriptorSet makeDescriptorSet(const Device &dev,
                                  VkDescriptorPool pool,
                                  VkDescriptorSetLayout layout)
{
    VkDescriptorSetAllocateInfo alloc;
    alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc.pNext = nullptr;
    alloc.descriptorPool = pool;
    alloc.descriptorSetCount = 1;
    alloc.pSetLayouts = &layout;

    VkDescriptorSet desc_set;
    REQ_VK(dev.dt.allocateDescriptorSets(dev.hdl, &alloc, &desc_set));

    return desc_set;
}

template <typename T>
T divideRoundUp(T a, T b)
{
    static_assert(std::is_integral_v<T>);

    return (a + (b - 1)) / b;
}

VkDeviceSize alignOffset(VkDeviceSize offset, VkDeviceSize alignment)
{
    return divideRoundUp(offset, alignment) * alignment;
}

uint32_t getWorkgroupSize(uint32_t num_items)
{
    return divideRoundUp(num_items, VulkanConfig::compute_workgroup_size);
}

}
}
}
