#include <madrona/render/vk/dispatch.hpp>
#include <madrona/crash.hpp>

#include <string>

namespace madrona {
namespace render {
namespace vk {

static inline PFN_vkVoidFunction checkPtr(PFN_vkVoidFunction ptr,
                                          const std::string &name)
{
    if (!ptr) {
        FATAL("Failed to load vulkan function: %s\n", name.c_str());
    }

    return ptr;
}

InstanceDispatch::InstanceDispatch(VkInstance ctx,
                                   PFN_vkGetInstanceProcAddr get_inst_addr,
                                   bool support_present)
#include "dispatch_instance_impl.cpp"
{}

DeviceDispatch::DeviceDispatch(VkDevice ctx,
                               PFN_vkGetDeviceProcAddr get_dev_addr,
                               bool support_present, bool support_rt,
                               bool support_mem_export)
#include "dispatch_device_impl.cpp"
{}

}
}
}
