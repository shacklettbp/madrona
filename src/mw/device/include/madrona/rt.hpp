#pragma once

#include <madrona/math.hpp>

namespace madrona::render::rt {

// Some pixel types:
namespace pix {
using PixelRGBA8 = uint32_t;
using DepthF32 = float;
}

// This struct gets passed to the RT kernel entry.
struct TraceKernelEntry {
    PerspectiveCameraData *camData;
    uint32_t pixelX;
    uint32_t pixelY;
};

// This gets returned from the RT kernel.
struct TraceKernelResult {
    float depth;
    math::Vector3 color;
};
    
struct TraceResult {
    bool hit;
    float depth;
    uint32_t materialID;
    math::Vector3 normal;
    math::Vector2 uvs;
    render::InstanceData *instance;
    MeshBVH *meshBVH;
};

struct TraceInfo {
    uint32_t worldID;
    math::Vector3 rayOrigin;
    math::Vector3 rayDir;
    float minDepth;
    float maxDepth;
};

TraceResult traceRay(const TraceInfo &trace_info);


// May or may not do texture sampling depending on the material.
// The UVs are returned from `traceRay`.
math::Vector3 sampleMaterialColor(int32_t mat_idx,
                                  MeshBVH *mesh_bvh,
                                  math::Vector2 uvs);

}

#define MADRONA_BUILD_RT_ENTRY(FuncT, ArchetypeT...)      \
    static_assert(alignof(::madrona::rt::RTEntry<         \
                FuncT, ArchetypeT, __VA_ARGS__>) == 16);
