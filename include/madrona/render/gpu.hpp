#pragma once

#include <madrona/types.hpp>
#include <madrona/render/platform.hpp>

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

struct CommandBuffer {
    platform::CommandBuffer hdl;
};

class GPU {
public:
    GPU();

    TextureHandle makeTex2D(int32_t width, int32_t height, TexFormat fmt);
    BufferHandle makeBuffer(int32_t num_bytes);

    void destroyTex2D(TextureHandle hdl);
    void destroyBuffer(BufferHandle hdl);

    void debugFullBarrier(CommandBuffer cmd);

    void submit(const RenderGraph &graph);

    inline platform::GPU & platform();

private:
    platform::GPU hdl_;
};

}

#include "gpu.inl"
