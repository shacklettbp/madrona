#pragma once

#include <array>
#include <optional>

#include <madrona/heap_array.hpp>
#include <madrona/span.hpp>
#include <madrona/optional.hpp>

#include "dispatch.hpp"

namespace madrona::render::vk {

using DeviceUUID = std::array<uint8_t, VK_UUID_SIZE>;

struct DeviceState {
public:
    uint32_t gfxQF;
    uint32_t computeQF;
    uint32_t transferQF;

    const VkPhysicalDevice phy;
    const VkDevice hdl;
    const DeviceDispatch dt;

    uint32_t numGraphicsQueues;
    uint32_t numComputeQueues;
    uint32_t numTransferQueues;
    bool rtAvailable;

    DeviceState(uint32_t gfx_qf, uint32_t compute_qf, uint32_t transfer_qf,
                uint32_t num_gfx_queues, uint32_t num_compute_queues,
                uint32_t num_transfer_queues, bool rt_available,
                VkPhysicalDevice phy_dev, VkDevice dev,
                DeviceDispatch &&dispatch_table);
    ~DeviceState();

    DeviceState(const DeviceState &) = delete;
    DeviceState(DeviceState &&) = default;
};

struct InstanceInitializer;

struct InstanceState {
public:
    VkInstance hdl;
    InstanceDispatch dt;

    InstanceState(PFN_vkGetInstanceProcAddr get_inst_addr,
                  bool enable_validation,
                  bool need_present,
                  Span<const char *const> extra_exts);
    ~InstanceState();

    InstanceState(const InstanceState &) = delete;
    InstanceState(InstanceState &&);

    DeviceState makeDevice(
        const DeviceUUID &uuid,
        uint32_t desired_gfx_queues,
        uint32_t desired_compute_queues,
        uint32_t desired_transfer_queues,
        Optional<VkSurfaceKHR> present_surface) const;

private:
    InstanceState(InstanceInitializer init, bool need_present);

    const VkDebugUtilsMessengerEXT debug_;
    void *loader_handle_;

    VkPhysicalDevice findPhysicalDevice(const DeviceUUID &uuid) const;
};

}
