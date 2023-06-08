#pragma once

#include <madrona/math.hpp>
#include <madrona/span.hpp>
#include <madrona/heap_array.hpp>

#include <string>

namespace madrona::render {

namespace refl {

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

struct SPIRV {
    HeapArray<EntryPoint> entryPoints;
    HeapArray<Binding> bindings;
    HeapArray<DescriptorSet> descriptorSets;
};

}

struct SPIRVShader {
    HeapArray<uint32_t> bytecode;
    refl::SPIRV reflectionInfo;
};

struct MTLShader {
    HeapArray<char> bytecode;
};

class ShaderCompiler {
public:
    ShaderCompiler();
    ~ShaderCompiler();

    SPIRVShader compileHLSLFileToSPV(
        const char *path,
        Span<const char *> include_dirs,
        Span<const char *> defines);

    MTLShader compileHLSLFileToMTL(
        const char *path,
        Span<const char *> include_dirs,
        Span<const char *> defines);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
