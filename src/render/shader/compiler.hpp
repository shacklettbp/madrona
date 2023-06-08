#pragma once

#include <madrona/math.hpp>
#include <madrona/span.hpp>
#include <madrona/heap_array.hpp>

#include <string>
#include <vector>

namespace madrona::render {

enum class ShaderStage {
    Vertex,
    Fragment,
    Compute,
    Mesh,
    Amplification,
};

struct SPIRVShader {
    HeapArray<uint32_t> bytecode;
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
        const char *entry_point,
        ShaderStage stage,
        Span<const char *> include_dirs,
        Span<const char *> defines);

    MTLShader compileHLSLFileToMTL(
        const char *path,
        const char *entry_point,
        ShaderStage stage,
        Span<const char *> include_dirs,
        Span<const char *> defines);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
