#pragma once

#include <madrona/heap_array.hpp>
#include <string>

namespace madrona::render {

enum class ShaderStage : uint32_t {
    Vertex = 1 << 0,
    Fragment = 1 << 1,
    Compute = 1 << 2,
    Mesh = 1 << 3,
    Amplification = 1 << 4,
};

namespace refl {

struct EntryPoint {
    std::string name;
    ShaderStage stage;
};

enum class BindingType : uint32_t {
    None = 0,
    Sampler = 1,
    Texture = 2,
    UniformBuffer = 3,
    StorageBuffer = 4,
    StorageImage = 5,
    AccelerationStructure = 6,
    NumBindingTypes = AccelerationStructure,
};

struct Binding {
    uint32_t id;
    BindingType type;
    uint32_t numResources;
    uint32_t stageUsage;
};

struct DescriptorSet {
    uint32_t id;
    uint32_t bindingOffset;
    uint32_t numBindings;
};

struct ShaderInfo {
    HeapArray<EntryPoint> entryPoints;
};

struct SPIRV : ShaderInfo {
    HeapArray<Binding> bindings;
    HeapArray<DescriptorSet> descriptorSets;
};

}

}
