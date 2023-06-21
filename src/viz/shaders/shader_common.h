#ifndef MADRONA_VIEWER_SHADER_COMMON_H_INCLUDED
#define MADRONA_VIEWER_SHADER_COMMON_H_INCLUDED

struct CullPushConst {
    uint32_t numInstances;
};

struct DrawPushConst {
    uint32_t viewIdx;
};

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
    int32_t materialIndex;
    int32_t pad[3];
};

struct MaterialData {
    // For now, just a color
    float4 color;
};

struct ObjectData {
    int32_t meshOffset;
    int32_t numMeshes;
};

struct PackedInstanceData {
    float4 data[3];
};

struct EngineInstanceData {
    float3 position;
    float4 rotation;
    float3 scale;
    int32_t objectID;
};

struct PackedViewData {
    float4 data[3];
};

struct ShadowViewData {
    float4x4 viewProjectionMatrix;
};

struct DirectionalLight {
    float4 lightDir;
    float4 color;
};

struct PerspectiveCameraData {
    float3 pos;
    float4 rot;
    float xScale;
    float yScale;
    float zNear;
};

struct DrawCmd {
    uint32_t indexCount;
    uint32_t instanceCount;
    uint32_t firstIndex;
    int32_t vertexOffset;
    uint32_t firstInstance;
};

struct DrawMaterialData {
    int32_t materialIdx;
};

#if 0
struct PackedDrawInstanceData {
    float4 packed[5];
};

struct DrawInstanceData {
    float3x3 toViewRot;
    float3 toViewTranslation;
    float3 objScale;
    int32_t viewIdx;
    float2 projScale;
    float projZNear;
};
#endif

#endif
