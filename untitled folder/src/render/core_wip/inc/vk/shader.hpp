#pragma once

#include <madrona/heap_array.hpp>

#include <madrona/render/vk/device.hpp>
#include <madrona/render/reflection.hpp>

// Issues:
// In metal, there are 2 argument buffers, one for vertex shader
// and one for fragment shader. Compute shader argument buffer
// maps 1 - 1.
//
// In Vulkan, we want to allocate all descriptors in one shot from one
// pool.

namespace madrona::render::vk {

using ParamBlock = VkDescriptorSet;

using RasterParamBlock = ParamBlock;
using ComputeParamBlock = ParamBlock;

using DescriptorTypeCounts =
    std::array<int32_t, (uint32_t)refl::NumBindingTypes>;

struct ParamBlockDesc {
    VkDescriptorSetLayout layout;
    DescriptorTypeCounts typeCounts;
};

struct ParamBlockStore {
    VkDescriptorPool pool;

    void destroy(Device &dev);
};

class ParamBlockBuilder {
public:
    ParamBlockBuilder();

    inline void addParamBlock(const ParamBlockDesc &desc, CountT num_blocks);

    ParamBlockStore build(ParamBlock *blocks_out);

private:
    DescriptorTypeCounts type_counts_;
};

class Shader {
public:
    Shader(Device &dev, StackAlloc &tmp_alloc,
           void *ir, CountT num_ir_bytes,
           const refl::ShaderInfo &refl_info);

    inline ParamBlockDesc getParamBlockDesc(CountT block_idx);

    void destroy(Device &dev);

private:
    inline Shader(Device &dev, void *ir, CountT num_ir_bytes,
                  const refl::SPIRV &refl_info);

    VkShaderModule shader_;
    HeapArray<VkDescriptorSetLayout> set_layouts_;
    HeapArray<DescriptorTypeCounts> set_type_counts_;
};

}

#include "shader.inl"
