#pragma once

#include <madrona/optional.hpp>
#include <madrona/span.hpp>

#include <madrona/render/vk/device.hpp>

namespace madrona::render::vk {

using DeviceID = std::array<uint8_t, VK_UUID_SIZE>;

class Backend {
public:
    Backend(void (*vk_entry_fn)(),
            bool enable_validation,
            bool headless,
            Span<const char *const> extra_vk_exts = {});

    Backend(const Backend &) = delete;
    Backend(Backend &&);
    ~Backend();

    Device initDevice(const DeviceID &gpu_id,
        Optional<VkSurfaceKHR> present_surface =
            Optional<VkSurfaceKHR>::none());

    VkInstance hdl;
    InstanceDispatch dt;

private:
    struct Init;
    inline Backend(Init init, bool headless);

    const VkDebugUtilsMessengerEXT debug_;
    void *loader_handle_;

    VkPhysicalDevice findPhysicalDevice(const DeviceID &id) const;
};

}
