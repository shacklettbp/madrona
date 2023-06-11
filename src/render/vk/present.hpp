#pragma once

#include <madrona/math.hpp>
#include <madrona/heap_array.hpp>
#include <madrona/render/vk/backend.hpp>

#include "utils.hpp"

namespace madrona {
namespace render {
namespace vk {

struct Swapchain {
    VkSwapchainKHR hdl;
    uint32_t width;
    uint32_t height;
};

struct Window {
    void *hdl;
};

class PresentationState {
public:
    static void (*init())();
    static void cleanup();

    static Window makeWindow(uint32_t width, uint32_t height);
    static HeapArray<const char *> getInstanceExtensions(
        const Window &window);

    static VkSurfaceKHR makeSurface(const Backend &backend,
                                    const Window &window);

    PresentationState(const Backend &backend,
                      const Device &dev,
                      Window &&window,
                      VkSurfaceKHR surface,
                      uint32_t qf_idx,
                      uint32_t num_frames_inflight,
                      bool need_immediate);
    ~PresentationState();

    void processInputs();

    void forceTransition(const Device &dev,
                         const GPURunUtil &gpu_run);

    uint32_t acquireNext(const Device &dev,
                         VkSemaphore signal_sema);

    VkImage getImage(uint32_t idx) const;
    uint32_t numSwapchainImages() const;

    void present(const Device &dev, uint32_t swapchain_idx,
                 VkQueue present_queue,
                 uint32_t num_wait_semas,
                 const VkSemaphore *wait_semas);

private:
    const Backend *backend_;
    const Device *dev_;
    Window window_;
    VkSurfaceKHR surface_;
    Swapchain swapchain_;
    HeapArray<VkImage> swapchain_imgs_;
};

}
}
}
