#ifndef MADRONA_VK_SHADER_COMMON_H_INCLUDED
#define MADRONA_VK_SHADER_COMMON_H_INCLUDED

#define NUM_SUBGROUPS (8)
#define SUBGROUP_SIZE (32)
#define WORKGROUP_SIZE (256)
#define LOCAL_WORKGROUP_X (32)
#define LOCAL_WORKGROUP_Y (8)
#define LOCAL_WORKGROUP_Z (1)

struct Vertex {
    vec3 position;
    vec3 normal;
    vec4 tangentAndSign;
    vec2 uv;
};

struct PackedVertex {
    vec4 data[2];
};

struct Camera {
    vec3 origin;
    vec3 view;
    vec3 up;
    vec3 right;
    float rightScale;
    float upScale;
};

struct PackedCamera {
    vec4 rotation;
    vec4 posAndTanFOV;
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
    uint32_t worldID;
    uint32_t pad[3];
};

#endif
