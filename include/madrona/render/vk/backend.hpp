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

    class LoaderLib {
    public:
        LoaderLib(void *lib, const char *env_str);
        LoaderLib(const LoaderLib &) = delete;
        LoaderLib(LoaderLib &&o);
        ~LoaderLib();

        void (*getEntryFn() const)();

    private:
        void *lib_;
        const char *env_str_;
    };

    static LoaderLib loadLoaderLib();

    Device initDevice(CountT gpu_idx, Optional<VkSurfaceKHR> present_surface =
        Optional<VkSurfaceKHR>::none());

    Device initDevice(const DeviceID &gpu_id,
        Optional<VkSurfaceKHR> present_surface =
            Optional<VkSurfaceKHR>::none());

    VkInstance hdl;
    InstanceDispatch dt;

private:
    Device initDevice(VkPhysicalDevice phy, 
                      Optional<VkSurfaceKHR> present_surface);

    struct Init;
    inline Backend(Init init, bool headless);

    const VkDebugUtilsMessengerEXT debug_;

    VkPhysicalDevice findPhysicalDevice(const DeviceID &id) const;
};

}
