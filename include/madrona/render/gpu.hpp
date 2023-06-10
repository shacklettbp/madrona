#pragma once

#include <madrona/types.hpp>

namespace madrona::render {

class RenderGraph; 

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

class GPU {
public:
    TextureHandle makeTex2D(int32_t width, int32_t height, TexFormat fmt);
    BufferHandle makeBuffer(int32_t num_bytes);

    void submit(const RenderGraph &graph);

private:
};

}
