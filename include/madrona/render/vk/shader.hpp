#pragma once

#include <madrona/heap_array.hpp>
#include <madrona/render/vk/shader.hpp>

namespace madrona::render::vk {

struct ParamBlock {
    VkDescriptorSet set;
};

class Shader {
public:

    ParamBlock makeParamBlock(CountT block_idx);

private:
    HeapArray<VkShaderModule> shaders_;
    HeapArray<VkDescriptorSetLayout> layouts_;
    HeapArray<HeapArray<VkDescriptorPoolSize>> base_pool_sizes_;
};

}
