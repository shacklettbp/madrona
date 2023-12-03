#pragma once

#include <madrona/math.hpp>
#include <madrona/span.hpp>
#include <madrona/stack_alloc.hpp>
#include <madrona/render/vk/device.hpp>
#include <madrona/render/shader_compiler.hpp>


#include <string>
#include <vector>

#include "shader.hpp"

namespace madrona {
namespace render {
namespace vk {

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
    PipelineShaders(const Device &dev,
                    StackAlloc &tmp_alloc,
                    Span<const SPIRVShader> shaders,
                    Span<const BindingOverride> binding_overrides,
                    VkShaderStageFlags extra_stages = 0);
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

    inline uint32_t getLayoutCount() const
    {
        return layouts_.size();
    }

    VkDescriptorPool makePool(uint32_t set_id, uint32_t max_sets) const;

private:
    const Device &dev;
    std::vector<VkShaderModule> shaders_;
    std::vector<VkDescriptorSetLayout> layouts_;
    std::vector<std::vector<VkDescriptorPoolSize>> base_pool_sizes_;
};

}
}
}
