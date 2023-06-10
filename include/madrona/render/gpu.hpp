#pragma once

namespace madrona::render {

struct ResourceHandle {
    uint32_t id;
    uint32_t gen;
};

struct TextureHandle : ResourceHandle {};
struct BufferHandle : ResourceHandle {};

struct Texture2DDesc {
};

struct BufferDesc {
    uint32_t numBytes;
};

class GPU {
public:


private:
};

}
