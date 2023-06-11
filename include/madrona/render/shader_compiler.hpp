#pragma once

#include <madrona/render/reflection.hpp>

namespace madrona::render {

struct SPIRVShader {
    HeapArray<uint32_t> bytecode;
    refl::SPIRV reflectionInfo;
};

struct MTLShader {
    HeapArray<char> bytecode;
};

class ShaderCompiler {
public:
    MADRONA_IMPORT ShaderCompiler();
    MADRONA_IMPORT ~ShaderCompiler();

    struct MacroDefn {
        const char *name;
        const char *value;
    };

    MADRONA_IMPORT SPIRVShader compileHLSLFileToSPV(
        const char *path,
        Span<const char *const> include_dirs,
        Span<const MacroDefn> macro_defns);

#ifdef MADRONA_APPLE
    MADRONA_IMPORT MTLShader compileHLSLFileToMTL(
        const char *path,
        Span<const char *const> include_dirs,
        Span<const MacroDefn> macro_defns);
#endif

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
