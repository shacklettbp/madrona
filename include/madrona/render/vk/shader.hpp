#pragma once

#include <madrona/heap_array.hpp>

#include <madrona/render/vk/device.hpp>

namespace madrona::render::vk {

struct ParamBlock {
    VkDescriptorSet set;
};

using RasterParamBlock = ParamBlock;
using ComputeParamBlock = ParamBlock;

class Shader {
public:
    Shader(Device &dev, void *ir, CountT num_ir_bytes,
           const refl::ShaderInfo &reflection_base);

    inline RasterParamBlock makeRasterParamBlock(CountT block_idx);
    inline ComputeParamBlock makeComputeParamBlock(CountT block_idx);

    void destroy(Device &dev);

private:
    inline Shader(Device &dev, const refl::SPIRV &refl_info);

    ParamBlock makeParamBlock(CountT block_idx);

    VkShaderModule shader_;
    HeapArray<VkDescriptorSetLayout> layouts_;
    HeapArray<HeapArray<VkDescriptorPoolSize>> base_pool_sizes_;
};

}

#include "shader.inl"
