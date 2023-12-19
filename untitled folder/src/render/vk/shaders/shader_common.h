#ifndef MADRONA_VK_SHADER_COMMON_H_INCLUDED
#define MADRONA_VK_SHADER_COMMON_H_INCLUDED

#define NUM_SUBGROUPS (8)
#define SUBGROUP_SIZE (32)
#define WORKGROUP_SIZE (256)
#define LOCAL_WORKGROUP_X (32)
#define LOCAL_WORKGROUP_Y (8)
#define LOCAL_WORKGROUP_Z (1)

struct Vertex {
    float3 position;
    float3 normal;
    float4 tangentAndSign;
    float2 uv;
};

struct PackedVertex {
    float4 data[2];
};

struct Camera {
    float3 origin;
    float3 view;
    float3 up;
    float3 right;
    float rightScale;
    float upScale;
};

struct PackedCamera {
    float4 rotation;
    float4 posAndTanFOV;
};

struct RTPushConstant {
    uint32_t frameCounter;
};

struct MeshData {
    uint32_t vertexOffset;
    uint32_t indexOffset;
};

struct ObjectData {
    uint64_t geoAddr;
};

struct ViewData {
    PackedCamera cam;
};

#endif
