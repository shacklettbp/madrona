#pragma once

#include <madrona/heap_array.hpp>
#include <string>

namespace madrona::render::refl {

enum class Stage : uint32_t {
    Vertex = 1 << 0,
    Fragment = 1 << 1,
    Compute = 1 << 2,
    Mesh = 1 << 3,
    Amplification = 1 << 4,
};

struct EntryPoint {
    std::string name;
    Stage stage;
};

enum class BindingType {
    None,
    Sampler,
    Texture,
    UniformBuffer,
    StorageBuffer,
    AccelerationStructure,
};

struct Binding {
    uint32_t id;
    BindingType type;
    uint32_t numResources;
    uint32_t stageUsage;
};

struct DescriptorSet {
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
