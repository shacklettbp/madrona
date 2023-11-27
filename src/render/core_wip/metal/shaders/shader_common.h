#ifndef MADRONA_BATCHRENDERER_METAL_SHADER_COMMON_H_INCLUDED
#define MADRONA_BATCHRENDERER_METAL_SHADER_COMMON_H_INCLUDED

#ifndef __METAL_VERSION__

struct command_buffer : MTL::ResourceID {
    inline command_buffer(MTL::ResourceID v)
        : ResourceID(v)
    {}
};

#endif

#if __METAL_VERSION__
#define MADRONA_METAL_CONST_PTR(Type) constant Type *
#define MADRONA_METAL_DEV_PTR(Type) device Type *
#else
#define MADRONA_METAL_CONST_PTR(Type) uint64_t
#define MADRONA_METAL_DEV_PTR(Type) uint64_t
#endif

namespace consts {

#ifdef __METAL_VERSION__
constant
#else
constexpr inline
#endif
    int32_t threadsPerInstance = 4;

}

struct Vertex {
    float3 position;
    float3 normal;
    float4 tangentAndSign;
    float2 uv;
};

struct PackedVertex {
    float4 data[2];
};

struct MeshData {
    int32_t vertexOffset;
    int32_t numVertices;
    int32_t indexOffset;
    int32_t numIndices;
};

struct ObjectData {
    int32_t meshOffset;
    int32_t numMeshes;
};

struct alignas(16) InstanceData {
    packed_float3 position;
    packed_float4 rotation;
    packed_float3 scale;
    int32_t objectID;
    int32_t worldID;
};

struct alignas(16) PerspectiveCameraData {
    packed_float3 position;
    packed_float4 rotation;
    float xScale;
    float yScale;
    float zNear;
    uint32_t pad[2];
};

struct alignas(16) DrawInstanceData {
    packed_float4 packed[5];
    float3x3 toViewRot;
    float3 toViewTranslation;
    float3 objScale;
    int32_t viewIdx;
    float projXScale;
    float projYScale;
    float projZNear;
};

struct AssetsArgBuffer {
    MADRONA_METAL_CONST_PTR(PackedVertex) vertices;
    MADRONA_METAL_CONST_PTR(uint32_t) indices;
    MADRONA_METAL_CONST_PTR(MeshData) meshes;
    MADRONA_METAL_CONST_PTR(ObjectData) objects;
};

struct DrawICBArgBuffer {
    command_buffer hdl;
};

struct RenderArgBuffer {
    MADRONA_METAL_DEV_PTR(DrawInstanceData) drawInstances;
    MADRONA_METAL_DEV_PTR(atomic_int) numDraws;
    MADRONA_METAL_CONST_PTR(InstanceData) engineInstances;
    MADRONA_METAL_CONST_PTR(PerspectiveCameraData) views;
    MADRONA_METAL_CONST_PTR(uint32_t) numViews;
    uint32_t numMaxViewsPerWorld;
};

#endif
