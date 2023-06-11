#pragma once

#include <vulkan/vulkan_core.h>
#include "core.hpp"

#include <madrona/math.hpp>
#include <madrona/span.hpp>

#include <string>
#include <vector>

#include <madrona/render/shader_compiler.hpp>

namespace madrona {
namespace render {
namespace vk {

namespace shader {
using float4 = madrona::math::Vector4;
using float3 = madrona::math::Vector3;
using float2 = madrona::math::Vector2;
#include "shaders/shader_common.h"
}

namespace VulkanConfig {

constexpr uint32_t compute_workgroup_size = WORKGROUP_SIZE;
constexpr uint32_t localWorkgroupX = LOCAL_WORKGROUP_X;
constexpr uint32_t localWorkgroupY = LOCAL_WORKGROUP_Y;
constexpr uint32_t localWorkgroupZ = LOCAL_WORKGROUP_Z;
constexpr uint32_t numSubgroups = NUM_SUBGROUPS;
constexpr uint32_t subgroupSize = SUBGROUP_SIZE;

}

struct BindingOverride {
    uint32_t setID;
    uint32_t bindingID;
    VkSampler sampler;
    uint32_t descriptorCount;
    VkDescriptorBindingFlags flags;
};

class PipelineShaders {
public:
    PipelineShaders(const DeviceState &dev,
                    const SPIRVShader &shader,
                    Span<const BindingOverride> binding_overrides);
    PipelineShaders(const PipelineShaders &) = delete;
    PipelineShaders(PipelineShaders &&) = default;
    ~PipelineShaders();

    inline VkShaderModule getShader(uint32_t idx) const
    {
        return shaders_[idx];
    }

    inline VkDescriptorSetLayout getLayout(uint32_t idx) const
    {
        return layouts_[idx];
    }

    VkDescriptorPool makePool(uint32_t set_id, uint32_t max_sets) const;

private:
    const DeviceState &dev;
    std::vector<VkShaderModule> shaders_;
    std::vector<VkDescriptorSetLayout> layouts_;
    std::vector<std::vector<VkDescriptorPoolSize>> base_pool_sizes_;
};

}
}
}
