#pragma once

#include <madrona/window.hpp>

#include <madrona/render/vk/backend.hpp>
#include <madrona/render/vk/window.hpp>

#include "vk/utils.hpp"

namespace madrona::render::vk {

struct Swapchain {
    VkSwapchainKHR hdl;
    uint32_t width;
    uint32_t height;
};

class PresentationState {
public:
    static VkSurfaceFormatKHR selectSwapchainFormat(
        const Backend &backend,
        VkPhysicalDevice phy,
        VkSurfaceKHR surface);

    PresentationState(const Backend &backend,
                      const Device &dev,
                      const RenderWindow *window,
                      uint32_t num_frames_inflight,
                      bool need_immediate);

    void destroy(const Device &dev);

    void forceTransition(const Device &dev,
                         const QueueState &present_queue,
                         uint32_t qf_idx);

    uint32_t acquireNext(const Device &dev,
                         VkSemaphore signal_sema,
                         bool &need_resize);

    VkImage getImage(uint32_t idx) const;
    uint32_t numSwapchainImages() const;

    void present(const Device &dev, uint32_t swapchain_idx,
                 const QueueState &present_queue,
                 uint32_t num_wait_semas,
                 const VkSemaphore *wait_semas,
                 bool &need_resize);

    void resize(const Backend &backend,
                const Device &dev,
                const RenderWindow *window,
                uint32_t num_frames_inflight,
                bool need_immediate);

private:
    Swapchain swapchain_;
    HeapArray<VkImage> swapchain_imgs_;
};

}
