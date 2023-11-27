#pragma once

namespace madrona::render {

enum class RenderContextFlags : uint32_t {
    None         = 0_u32,
    RenderBatch  = 1_u32 << 0,
    RenderViewer = 1_u32 << 1
};

inline RenderContextFlags & operator|=(RenderContextFlags &a,
                                       RenderContextFlags b);
inline RenderContextFlags operator|(RenderContextFlags a,
                                    RenderContextFlags b);
inline RenderContextFlags & operator&=(RenderContextFlags &a,
                                       RenderContextFlags b);
inline RenderContextFlags operator&(RenderContextFlags a,
                                    RenderContextFlags b);

}

#include "render_config.inl"
