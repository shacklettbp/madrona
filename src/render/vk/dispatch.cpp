#include "dispatch.hpp"
#include <iostream>
#include <cstdlib>

namespace madrona {
namespace render {
namespace vk {

static inline PFN_vkVoidFunction checkPtr(PFN_vkVoidFunction ptr,
                                          const std::string &name)
{
    if (!ptr) {
        std::cerr << name << " failed to load" << std::endl;
        exit(EXIT_FAILURE);
    }

    return ptr;
}

InstanceDispatch::InstanceDispatch(VkInstance ctx,
                                   PFN_vkGetInstanceProcAddr get_inst_addr,
                                   bool need_present)
#include "dispatch_instance_impl.cpp"
{}

DeviceDispatch::DeviceDispatch(VkDevice ctx,
                               PFN_vkGetDeviceProcAddr get_dev_addr,
                               bool need_present, bool need_rt)
#include "dispatch_device_impl.cpp"
{}

}
}
}
