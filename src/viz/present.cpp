#include <chrono>
#include "present.hpp"

namespace madrona::render::vk {

static VkSurfaceFormatKHR selectSwapchainFormat(const Backend &backend,
                                                VkPhysicalDevice phy,
                                                VkSurfaceKHR surface)
{
    uint32_t num_formats;
    REQ_VK(backend.dt.getPhysicalDeviceSurfaceFormatsKHR(
            phy, surface, &num_formats, nullptr));

    HeapArray<VkSurfaceFormatKHR> formats(num_formats);
    REQ_VK(backend.dt.getPhysicalDeviceSurfaceFormatsKHR(
            phy, surface, &num_formats, formats.data()));

    if (num_formats == 0) {
        FATAL("Zero swapchain formats");
    }

    // FIXME
    for (VkSurfaceFormatKHR format : formats) {
        if (format.format == VK_FORMAT_B8G8R8A8_SRGB &&
            format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return format;
        }
    }

    return formats[0];
}

static VkPresentModeKHR selectSwapchainMode(const Backend &backend,
                                            VkPhysicalDevice phy,
                                            VkSurfaceKHR surface,
                                            bool need_immediate)
{
    uint32_t num_modes;
    REQ_VK(backend.dt.getPhysicalDeviceSurfacePresentModesKHR(
            phy, surface, &num_modes, nullptr));

    HeapArray<VkPresentModeKHR> modes(num_modes);
    REQ_VK(backend.dt.getPhysicalDeviceSurfacePresentModesKHR(
            phy, surface, &num_modes, modes.data()));

    for (VkPresentModeKHR mode : modes) {
        if (need_immediate && mode == VK_PRESENT_MODE_IMMEDIATE_KHR) {
            return mode;
        } else if (!need_immediate && mode == VK_PRESENT_MODE_FIFO_KHR) {
            return mode;
        }
    }

    if (!need_immediate) {
        return modes[0];
    } else {
        FATAL("Could not find immediate swapchain");
    }
}

static Swapchain makeSwapchain(const Backend &backend,
                               const Device &dev,
                               const RenderWindow *window,
                               uint32_t num_frames_inflight,
                               bool need_immediate)
{
    VkSurfaceFormatKHR format =
        selectSwapchainFormat(backend, dev.phy, window->surface);
    VkPresentModeKHR mode =
        selectSwapchainMode(backend, dev.phy, window->surface, need_immediate);

    VkSurfaceCapabilitiesKHR caps;
    REQ_VK(backend.dt.getPhysicalDeviceSurfaceCapabilitiesKHR(
            dev.phy, window->surface, &caps));

    VkExtent2D swapchain_size = caps.currentExtent;
    if (swapchain_size.width == UINT32_MAX &&
        swapchain_size.height == UINT32_MAX) {
        glfwGetWindowSize(window->hdl, (int *)&swapchain_size.width,
                          (int *)&swapchain_size.height);

        swapchain_size.width = std::max(caps.minImageExtent.width,
                                   std::min(caps.maxImageExtent.width,
                                       swapchain_size.width));

        swapchain_size.height = std::max(caps.minImageExtent.height,
                                    std::min(caps.maxImageExtent.height,
                                        swapchain_size.height));
    }

    uint32_t num_requested_images =
        std::max(caps.minImageCount + 1, num_frames_inflight);
    if (caps.maxImageCount != 0 && num_requested_images > caps.maxImageCount) {
        num_requested_images = caps.maxImageCount;
    }

    VkSwapchainCreateInfoKHR swapchain_info;
    swapchain_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapchain_info.pNext = nullptr;
    swapchain_info.flags = 0;
    swapchain_info.surface = window->surface;
    swapchain_info.minImageCount = num_requested_images;
    swapchain_info.imageFormat = format.format;
    swapchain_info.imageColorSpace = format.colorSpace;
    swapchain_info.imageExtent = swapchain_size;
    swapchain_info.imageArrayLayers = 1;
    swapchain_info.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    swapchain_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapchain_info.queueFamilyIndexCount = 0;
    swapchain_info.pQueueFamilyIndices = nullptr;
    swapchain_info.preTransform = caps.currentTransform;
    swapchain_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapchain_info.presentMode = mode;
    swapchain_info.clipped = VK_TRUE;
    swapchain_info.oldSwapchain = VK_NULL_HANDLE;

    using namespace std::chrono;

    VkSwapchainKHR swapchain;
    REQ_VK(dev.dt.createSwapchainKHR(dev.hdl, &swapchain_info, nullptr,
                                     &swapchain));

    return Swapchain {
        swapchain,
        swapchain_size.width,
        swapchain_size.height,
    };
}

static HeapArray<VkImage> getSwapchainImages(const Device &dev,
                                            VkSwapchainKHR swapchain)
{
    uint32_t num_images;
    REQ_VK(dev.dt.getSwapchainImagesKHR(dev.hdl, swapchain, &num_images,
                                        nullptr));

    HeapArray<VkImage> swapchain_images(num_images);
    REQ_VK(dev.dt.getSwapchainImagesKHR(dev.hdl, swapchain, &num_images,
                                        swapchain_images.data()));

    return swapchain_images;
}

PresentationState::PresentationState(const Backend &backend,
                                     const Device &dev,
                                     const RenderWindow *window,
                                     uint32_t num_frames_inflight,
                                     bool need_immediate)
    : swapchain_(makeSwapchain(backend, dev, window,
                               num_frames_inflight,
                               need_immediate)),
      swapchain_imgs_(getSwapchainImages(dev, swapchain_.hdl))
{
}

void PresentationState::destroy(const Device &dev)
{
    dev.dt.destroySwapchainKHR(dev.hdl, swapchain_.hdl, nullptr);
}

void PresentationState::forceTransition(const Device &dev,
    const QueueState &present_queue, uint32_t qf_idx)
{
    VkCommandPool tmp_pool = makeCmdPool(dev, qf_idx);
    VkCommandBuffer cmd = makeCmdBuffer(dev, tmp_pool);

    std::vector<VkImageMemoryBarrier> barriers;
    barriers.reserve(swapchain_imgs_.size());

    for (int i = 0; i < (int)swapchain_imgs_.size(); i++) {
        barriers.push_back({
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            nullptr,
            0,
            0,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED,
            swapchain_imgs_[i],
            { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
        });
    }

    VkCommandBufferBeginInfo begin_info {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    REQ_VK(dev.dt.beginCommandBuffer(cmd, &begin_info));

    dev.dt.cmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              0, 0, nullptr, 0, nullptr,
                              barriers.size(), barriers.data());

    REQ_VK(dev.dt.endCommandBuffer(cmd));

    VkSubmitInfo render_submit {};
    render_submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    render_submit.waitSemaphoreCount = 0;
    render_submit.pWaitSemaphores = nullptr;
    render_submit.pWaitDstStageMask = nullptr;
    render_submit.commandBufferCount = 1;
    render_submit.pCommandBuffers = &cmd;

    VkFence fence = makeFence(dev);

    present_queue.submit(dev, 1, &render_submit, fence);

    waitForFenceInfinitely(dev, fence);

    // FIXME, get an initialization pool / fence for stuff like this
    dev.dt.destroyFence(dev.hdl, fence, nullptr);
    dev.dt.destroyCommandPool(dev.hdl, tmp_pool, nullptr);
}

uint32_t PresentationState::acquireNext(const Device &dev,
                                        VkSemaphore signal_sema,
                                        bool &need_resize)
{
    uint32_t swapchain_idx;
    VkResult result = dev.dt.acquireNextImageKHR(dev.hdl, swapchain_.hdl,
                                                 0, signal_sema,
                                                 VK_NULL_HANDLE,
                                                 &swapchain_idx);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        need_resize = true;
    } else if (result == VK_SUBOPTIMAL_KHR) {
        need_resize = true;
    } else if (result != VK_SUCCESS) {
        assert(false);
    } else {
        need_resize = false;
    }

    return swapchain_idx;
}

VkImage PresentationState::getImage(uint32_t idx) const
{
    return swapchain_imgs_[idx];
}

uint32_t PresentationState::numSwapchainImages() const
{
    return swapchain_imgs_.size();
}

void PresentationState::present(const Device &dev, uint32_t swapchain_idx,
                                const QueueState &present_queue,
                                uint32_t num_wait_semas,
                                const VkSemaphore *wait_semas,
                                bool &need_resize)
{
    VkPresentInfoKHR present_info;
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.pNext = nullptr;
    present_info.waitSemaphoreCount = num_wait_semas;
    present_info.pWaitSemaphores = wait_semas;

    present_info.swapchainCount = 1;
    present_info.pSwapchains = &swapchain_.hdl;
    present_info.pImageIndices = &swapchain_idx;
    present_info.pResults = nullptr;

    bool submit_success = present_queue.presentSubmit(dev, &present_info);

    if (submit_success) {
        // If the present was successful, no need to resize
        need_resize = false;
    } else {
        // If the present failed, need to resize
        need_resize = true;
    }
}


void PresentationState::resize(const Backend &backend,
                               const Device &dev,
                               const RenderWindow *window,
                               uint32_t num_frames_inflight,
                               bool need_immediate)
{
    // Destroy the swapchain first
    destroy(dev);

    swapchain_ = makeSwapchain(backend, 
                               dev,
                               window,
                               num_frames_inflight,
                               need_immediate);

    swapchain_imgs_.clear();

    swapchain_imgs_ = getSwapchainImages(dev, swapchain_.hdl);
}

}
