#pragma once

#include <vulkan/vulkan_core.h>
#include "core.hpp"

#include <madrona/math.hpp>

#include <string>
#include <vector>

namespace madrona {
namespace render {
namespace vk {

namespace shader {
using float3 = madrona::math::Vector3;
using float2 = madrona::math::Vector2;
#include "shaders/shader_common.h"
}

using shader::Vertex;
using shader::Mesh;
using shader::Object;

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

class ShaderPipeline {
public:
    ShaderPipeline(const DeviceState &dev,
                   const std::vector<std::string> &shader_paths,
                   const std::vector<BindingOverride> &binding_overrides,
                   const std::vector<std::string> &defines,
                   const char *shader_dir);
    ShaderPipeline(const ShaderPipeline &) = delete;
    ShaderPipeline(ShaderPipeline &&) = default;
    ~ShaderPipeline();

    static void initCompiler();

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
