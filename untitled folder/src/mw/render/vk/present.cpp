#include "present.hpp"

#include <madrona/heap_array.hpp>

#include <cstdlib>
#include <cstring>
#include <iostream>

#include <SDL.h>
#include <SDL_vulkan.h>

namespace madrona {
namespace render {
namespace vk {

static inline int checkSDL(int res, const char *msg)
{
    if (res < 0) {
        FATAL("%s: %s\n", msg, SDL_GetError());
    }

    return res;
}

template <typename T>
static inline T *checkSDLPointer(T *ptr, const char *msg)
{
    if (ptr == nullptr) {
        FATAL("%s: %s\n", msg, SDL_GetError());
    }

    return ptr;
}

#define REQ_SDL(expr) checkSDL((expr), #expr)
#define NONNULL_SDL(expr) checkSDLPointer((expr), #expr)

void (*PresentationState::init())()
{
    REQ_SDL(SDL_Init(SDL_INIT_VIDEO));
    REQ_SDL(SDL_Vulkan_LoadLibrary(nullptr));

    return (void (*)())SDL_Vulkan_GetVkGetInstanceProcAddr();
}

void PresentationState::cleanup()
{
    SDL_Vulkan_UnloadLibrary();
    SDL_QuitSubSystem(SDL_INIT_VIDEO);
    SDL_Quit();
}

Window PresentationState::makeWindow(uint32_t width, uint32_t height)
{
    uint32_t flags = SDL_WINDOW_ALLOW_HIGHDPI | SDL_WINDOW_BORDERLESS |
        SDL_WINDOW_VULKAN;

    SDL_Window *hdl = NONNULL_SDL(SDL_CreateWindow("madrona",
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height,
        flags));

    return Window {
        hdl,
    };
}

HeapArray<const char *> PresentationState::getInstanceExtensions(
    const Window &window)
{
    unsigned int count;
    if (!SDL_Vulkan_GetInstanceExtensions((SDL_Window *)window.hdl,
                                          &count, nullptr)) {
        FATAL("Failed to get vulkan instance extension count from SDL");
    }

    HeapArray<const char *> extensions(count);
    if (!SDL_Vulkan_GetInstanceExtensions((SDL_Window *)window.hdl,
                                          &count, extensions.data())) {
        FATAL("Failed to get vulkan instance extensions from SDL");
    }

    return extensions;
}

VkSurfaceKHR PresentationState::makeSurface(const Backend &backend,
                                            const Window &window)
{
    VkSurfaceKHR surface;
    if (!SDL_Vulkan_CreateSurface((SDL_Window *)window.hdl,
                                  backend.hdl, &surface)) {
        FATAL("Failed to create vulkan window surface");
    }

    return surface;
}

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
                               void *window_hdl,
                               VkSurfaceKHR surface,
                               uint32_t qf_idx,
                               uint32_t num_frames_inflight,
                               bool need_immediate)
{
    SDL_Window *window = (SDL_Window *)window_hdl;

    // Need to include this call despite the platform specific check
    // earlier (pre surface creation), or validation layers complain
    VkBool32 surface_supported;
    REQ_VK(backend.dt.getPhysicalDeviceSurfaceSupportKHR(
            dev.phy, qf_idx, surface, &surface_supported));

    if (surface_supported == VK_FALSE) {
        FATAL("SDL surface doesn't support presentation");
    }

    VkSurfaceFormatKHR format = selectSwapchainFormat(backend, dev.phy, surface);
    VkPresentModeKHR mode = selectSwapchainMode(backend, dev.phy, surface,
                                                need_immediate);

    VkSurfaceCapabilitiesKHR caps;
    REQ_VK(backend.dt.getPhysicalDeviceSurfaceCapabilitiesKHR(
            dev.phy, surface, &caps));

    VkExtent2D swapchain_size = caps.currentExtent;
    if (swapchain_size.width == UINT32_MAX &&
        swapchain_size.height == UINT32_MAX) {
        int drawable_width, drawable_height;
        SDL_Vulkan_GetDrawableSize(window, &drawable_width, &drawable_height);

        swapchain_size.width = std::max(caps.minImageExtent.width,
                                   std::min(caps.maxImageExtent.width,
                                       (uint32_t)drawable_width));

        swapchain_size.height = std::max(caps.minImageExtent.height,
                                    std::min(caps.maxImageExtent.height,
                                        (uint32_t)drawable_height));
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
    swapchain_info.surface = surface;
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
                                     Window &&window,
                                     VkSurfaceKHR surface,
                                     uint32_t qf_idx,
                                     uint32_t num_frames_inflight,
                                     bool need_immediate)
    : backend_(&backend),
      dev_(&dev),
      window_(std::move(window)),
      surface_(surface),
      swapchain_(makeSwapchain(backend, dev, window_.hdl, surface_,
                               qf_idx, num_frames_inflight,
                               need_immediate)),
      swapchain_imgs_(getSwapchainImages(dev, swapchain_.hdl))
{}

PresentationState::~PresentationState()
{
    dev_->dt.destroySwapchainKHR(dev_->hdl, swapchain_.hdl, nullptr);
    backend_->dt.destroySurfaceKHR(backend_->hdl, surface_, nullptr);
    SDL_DestroyWindow((SDL_Window *)window_.hdl);
}

void PresentationState::processInputs()
{
    static std::array<SDL_Event, 1024> events;

    SDL_PumpEvents();

    int num_events;
    do {
        num_events = REQ_SDL(SDL_PeepEvents(events.data(), events.size(),
                             SDL_GETEVENT, SDL_FIRSTEVENT, SDL_LASTEVENT));
    } while (num_events == events.size());

    // Right now this function pretty much just exists to get
    // gnome to shut up about the window not responding
    (void)num_events;
}

void PresentationState::forceTransition(const Device &dev,
    const GPURunUtil &gpu_run)
{
    HeapArray<VkImageMemoryBarrier> barriers(swapchain_imgs_.size());

    for (int i = 0; i < (int)swapchain_imgs_.size(); i++) {
        barriers[i] = {
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
        };
    }

    gpu_run.begin(dev);

    dev.dt.cmdPipelineBarrier(gpu_run.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              0, 0, nullptr, 0, nullptr,
                              barriers.size(), barriers.data());

    gpu_run.submit(dev);
}

uint32_t PresentationState::acquireNext(const Device &dev,
                                        VkSemaphore signal_sema)
{
    uint32_t swapchain_idx;
    REQ_VK(dev.dt.acquireNextImageKHR(dev.hdl, swapchain_.hdl,
                                      0, signal_sema,
                                      VK_NULL_HANDLE,
                                      &swapchain_idx));

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
                                VkQueue present_queue,
                                uint32_t num_wait_semas,
                                const VkSemaphore *wait_semas)
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

    REQ_VK(dev.dt.queuePresentKHR(present_queue, &present_info));
}

}
}
}
