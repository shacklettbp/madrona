#pragma once

#include <madrona/span.hpp>
#include <madrona/render/gpu.hpp>
#include <madrona/render/reflection.hpp>

namespace madrona::render {

struct ParamBlock {
    backend::ParamBlock hdl;
};

class Shader {
public:
    inline Shader(GPU &gpu, Span<char> bytecode,
                  const refl::ShaderInfo &reflection_info);

    inline ParamBlock makeParamBlock(CountT block_idx);

    inline void destroy(GPU &gpu);

private:
    backend::Shader hdl_;
};

}

#include "shader.inl"
