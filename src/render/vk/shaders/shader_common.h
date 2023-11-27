#ifndef MADRONA_VK_SHADER_COMMON_H_INCLUDED
#define MADRONA_VK_SHADER_COMMON_H_INCLUDED

#define PREPARE_VIEW_WORKGROUP_SIZE (32)

#define NUM_SUBGROUPS (8)
#define SUBGROUP_SIZE (32)
#define WORKGROUP_SIZE (256)
#define LOCAL_WORKGROUP_X (32)
#define LOCAL_WORKGROUP_Y (8)
#define LOCAL_WORKGROUP_Z (1)

struct PackedPerspectiveCameraData {
    float4 position;
    float4 rotation;
    float xScale;
    float yScale;
    float zNear;
    int worldIDX;
    int pad;
};

struct PrepareViewPushConstant {
    uint numViews;
};

struct WorldInstanceInfo {
    uint offset;
    uint count;
};

struct PackedInstanceData {
    float3 position;
    float4 rotation;
    float3 scale;
    int objectID;
    int worldID;
};

struct DrawCommandInfo {

};

#endif
