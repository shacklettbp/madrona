#pragma once 

#include <madrona/render/common.hpp>
#include <madrona/render/vk/dispatch.hpp>

namespace madrona::render::vk {

class Device : public GPUDevice {
public:
    VkDevice hdl;
    DeviceDispatch dt;
    VkPhysicalDevice phy;

    uint32_t gfxQF;
    uint32_t computeQF;
    uint32_t transferQF;

    uint32_t numGraphicsQueues;
    uint32_t numComputeQueues;
    uint32_t numTransferQueues;
    bool rtAvailable;

    uint32_t maxNumLayersPerImage;
    uint32_t maxViewports;
    uint32_t maxImageDim;

    float timestampPeriod;

    Device(uint32_t gfx_qf, uint32_t compute_qf, uint32_t transfer_qf,
           uint32_t num_gfx_queues, uint32_t num_compute_queues,
           uint32_t num_transfer_queues, bool rt_available,
           uint32_t max_num_layers_per_img,
           uint32_t max_viewports,
           uint32_t max_image_dim,
           float timestamp_period,
           VkPhysicalDevice phy_dev, VkDevice dev,
           DeviceDispatch &&dispatch_table);

    Device(const Device &) = delete;
    Device(Device &&o);

    ~Device();
};

}
