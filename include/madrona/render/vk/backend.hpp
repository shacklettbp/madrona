#pragma once

#include <madrona/render/vk/device.hpp>
#include <madrona/render/vk/shader.hpp>

namespace madrona::render::vk {

using DeviceID = std::array<uint8_t, VK_UUID_SIZE>;

class Backend {
public:
    Backend(void (*vk_entry_fn)(),
            bool enable_validation,
            bool headless,
            Span<const char *const> extra_vk_exts = {});
    ~Backend();

    Backend(const Backend &) = delete;
    Backend(Backend &&);

    Device initDevice(const DeviceID &gpu_id,
        Optional<VkSurfaceKHR> present_surface =
            Optional<VkSurfaceKHR>::none());

    const VkInstance hdl;
    const InstanceDispatch dt;

private:
    struct Init;
    Backend(Init init, bool headless);

    const VkDebugUtilsMessengerEXT debug_;
    void *loader_handle_;

    VkPhysicalDevice findPhysicalDevice(const DeviceUUID &uuid) const;
};

}
