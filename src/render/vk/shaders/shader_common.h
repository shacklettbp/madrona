#ifndef MADRONA_VK_SHADER_COMMON_H_INCLUDED
#define MADRONA_VK_SHADER_COMMON_H_INCLUDED

#define NUM_SUBGROUPS (8)
#define SUBGROUP_SIZE (32)
#define WORKGROUP_SIZE (256)
#define LOCAL_WORKGROUP_X (16)
#define LOCAL_WORKGROUP_Y (16)
#define LOCAL_WORKGROUP_Z (1)

struct Vertex {
    float3 position;
    float3 normalTangentPacked;
    float2 uv;
};

struct Mesh {
    uint32_t vertexOffset;
    uint32_t indexOffset;
    uint32_t numTriangles;
};

struct Object {
    uint32_t meshOffset;
    uint32_t numMeshes;
};

#endif
