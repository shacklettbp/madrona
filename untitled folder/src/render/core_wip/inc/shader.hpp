#pragma once

#include <madrona/span.hpp>
#include <madrona/render/gpu.hpp>
#include <madrona/render/reflection.hpp>

namespace madrona::render {

struct RasterParamBlock {
    backend::RasterParamBlock hdl;
};

struct ComputeParamBlock {
    backend::ComputeParamBlock hdl;
};

class ParamBlockAllocator {
public:

private:
    backend::ParamBlockAllocator alloc_;
};

class Shader {
public:
    inline Shader(GPU &gpu, StackAlloc &tmp_alloc,
                  void *ir, CountT num_ir_bytes,
                  const refl::ShaderInfo &reflection_info);

    inline RasterParamBlock makeRasterParamBlock(CountT block_idx);
    inline ComputeParamBlock makeComputeParamBlock(CountT block_idx);

    inline void destroy(GPU &gpu);

private:
    backend::Shader hdl_;
};

}

#include "shader.inl"
