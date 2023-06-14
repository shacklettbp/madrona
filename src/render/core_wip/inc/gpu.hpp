#pragma once

#include <madrona/types.hpp>
#include <madrona/render/backend.hpp>

#include <madrona/render/shader.hpp>

namespace madrona::render {

class RenderGraph; 
class GPU; 

struct ResourceHandle {
    uint32_t id;
    uint32_t gen;
};

struct TextureHandle : ResourceHandle {};
struct BufferHandle : ResourceHandle {};

enum class TexFormat {
    RGBA8Unorm,
};

struct Texture2DDesc {
    int32_t width;
    int32_t height;
    TexFormat fmt;
};

struct BufferDesc {
    int32_t numBytes;
};

class RasterCmdList {
public:
    inline void setParamBlock(GPU &gpu, ParamBlock param_block);
    void debugFullBarrier(GPU &gpu);


    inline backend::RasterCmdList & hdl();

private:
    backend::RasterCmdList hdl_;
};

class ComputeCmdList {
public:
    inline void setParamBlock(GPU &gpu, ParamBlock param_block);
    void debugFullBarrier(GPU &gpu);

    inline backend::ComputeCmdList & hdl();

private:
    backend::ComputeCmdList hdl_;
};

class GPU {
public:
    inline GPU(backend::Backend &backend,
               const backend::DeviceID &dev_id);

    TextureHandle makeTex2D(int32_t width, int32_t height, TexFormat fmt);
    BufferHandle makeBuffer(int32_t num_bytes);

    void destroyTex2D(TextureHandle hdl);
    void destroyBuffer(BufferHandle hdl);

    inline backend::Device & hdl();

private:
    backend::Device dev_;
};

}

#include "gpu.inl"
