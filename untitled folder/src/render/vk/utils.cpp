#include "utils.hpp"

#include <cstdlib>
#include <iostream>

using namespace std;

namespace madrona {
namespace render {
namespace vk {

void GPURunUtil::begin(const Device &dev) const
{
    VkCommandBufferBeginInfo begin_info {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    REQ_VK(dev.dt.beginCommandBuffer(cmd, &begin_info));
}

void GPURunUtil::submit(const Device &dev) const
{
    REQ_VK(dev.dt.endCommandBuffer(cmd));

    VkSubmitInfo submit {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.waitSemaphoreCount = 0;
    submit.pWaitSemaphores = nullptr;
    submit.pWaitDstStageMask = nullptr;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;

    dev.dt.queueSubmit(queue, 1, &submit, fence);
    waitForFenceInfinitely(dev, fence);

    dev.dt.resetCommandPool(dev.hdl, pool, 0);
    resetFence(dev, fence);
}

int exportBinarySemaphore(const Device &dev, VkSemaphore semaphore)
{
    VkSemaphoreGetFdInfoKHR fd_info;
    fd_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    fd_info.pNext = nullptr;
    fd_info.semaphore = semaphore;
    fd_info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

    int fd;
    REQ_VK(dev.dt.getSemaphoreFdKHR(dev.hdl, &fd_info, &fd));

    return fd;
}

VkSampler makeImmutableSampler(const Device &dev,
                               VkSamplerAddressMode address_mode)
{
    VkSampler sampler;

    VkSamplerCreateInfo sampler_info;
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.pNext = nullptr;
    sampler_info.flags = 0;
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler_info.addressModeU = address_mode;
    sampler_info.addressModeV = address_mode;
    sampler_info.addressModeW = address_mode;
    sampler_info.mipLodBias = 0;
    sampler_info.anisotropyEnable = VK_TRUE;
    sampler_info.maxAnisotropy = 16.0;
    sampler_info.compareEnable = VK_FALSE;
    sampler_info.compareOp = VK_COMPARE_OP_ALWAYS;
    sampler_info.minLod = 0;
    sampler_info.maxLod = VK_LOD_CLAMP_NONE;
    sampler_info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    sampler_info.unnormalizedCoordinates = VK_FALSE;

    REQ_VK(dev.dt.createSampler(dev.hdl, &sampler_info, nullptr, &sampler));

    return sampler;
}

void printVkError(VkResult res, const char *msg)
{
#define ERR_CASE(val) \
    case VK_##val:    \
        cerr << #val; \
        break

    cerr << msg << ": ";
    switch (res) {
        ERR_CASE(NOT_READY);
        ERR_CASE(TIMEOUT);
        ERR_CASE(EVENT_SET);
        ERR_CASE(EVENT_RESET);
        ERR_CASE(INCOMPLETE);
        ERR_CASE(ERROR_OUT_OF_HOST_MEMORY);
        ERR_CASE(ERROR_OUT_OF_DEVICE_MEMORY);
        ERR_CASE(ERROR_INITIALIZATION_FAILED);
        ERR_CASE(ERROR_DEVICE_LOST);
        ERR_CASE(ERROR_MEMORY_MAP_FAILED);
        ERR_CASE(ERROR_LAYER_NOT_PRESENT);
        ERR_CASE(ERROR_EXTENSION_NOT_PRESENT);
        ERR_CASE(ERROR_FEATURE_NOT_PRESENT);
        ERR_CASE(ERROR_INCOMPATIBLE_DRIVER);
        ERR_CASE(ERROR_TOO_MANY_OBJECTS);
        ERR_CASE(ERROR_FORMAT_NOT_SUPPORTED);
        ERR_CASE(ERROR_FRAGMENTED_POOL);
        ERR_CASE(ERROR_UNKNOWN);
        ERR_CASE(ERROR_OUT_OF_POOL_MEMORY);
        ERR_CASE(ERROR_INVALID_EXTERNAL_HANDLE);
        ERR_CASE(ERROR_FRAGMENTATION);
        ERR_CASE(ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS);
        ERR_CASE(ERROR_SURFACE_LOST_KHR);
        ERR_CASE(ERROR_NATIVE_WINDOW_IN_USE_KHR);
        ERR_CASE(SUBOPTIMAL_KHR);
        ERR_CASE(ERROR_OUT_OF_DATE_KHR);
        ERR_CASE(ERROR_INCOMPATIBLE_DISPLAY_KHR);
        ERR_CASE(ERROR_VALIDATION_FAILED_EXT);
        ERR_CASE(ERROR_INVALID_SHADER_NV);
        ERR_CASE(ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT);
        ERR_CASE(ERROR_NOT_PERMITTED_EXT);
        ERR_CASE(ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT);
        default:
            cerr << "New vulkan error";
            break;
    }
    cerr << endl;
#undef ERR_CASE
}

}
}
}
