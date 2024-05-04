#include <madrona/render/vk/device.hpp>

#include <utility>

namespace madrona::render::vk {

Device::Device(uint32_t gfx_qf, uint32_t compute_qf, uint32_t transfer_qf,
               uint32_t num_gfx_queues, uint32_t num_compute_queues,
               uint32_t num_transfer_queues, bool rt_available,
               uint32_t max_num_layers_per_img,
               uint32_t max_viewports,
               uint32_t max_image_dim,
               float timestamp_period,
               VkPhysicalDevice phy_dev, VkDevice dev,
               DeviceDispatch &&dispatch_table)
    : GPUDevice(),
      hdl(dev),
      dt(std::move(dispatch_table)),
      phy(phy_dev),
      gfxQF(gfx_qf),
      computeQF(compute_qf),
      transferQF(transfer_qf),
      numGraphicsQueues(num_gfx_queues), 
      numComputeQueues(num_compute_queues),
      numTransferQueues(num_transfer_queues),
      rtAvailable(rt_available),
      maxNumLayersPerImage(max_num_layers_per_img),
      maxViewports(max_viewports),
      maxImageDim(max_image_dim),
      timestampPeriod(timestamp_period)
{}

Device::Device(Device &&o)
    : GPUDevice(),
      hdl(o.hdl),
      dt(std::move(o.dt)),
      phy(o.phy),
      gfxQF(o.gfxQF),
      computeQF(o.computeQF),
      transferQF(o.transferQF),
      numGraphicsQueues(o.numGraphicsQueues), 
      numComputeQueues(o.numComputeQueues),
      numTransferQueues(o.numTransferQueues),
      rtAvailable(o.rtAvailable)
{
    o.hdl = VK_NULL_HANDLE;
}

Device::~Device()
{
    if (hdl != VK_NULL_HANDLE) {
        dt.destroyDevice(hdl, nullptr);
    }
}

}
