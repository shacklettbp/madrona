#include "shader_common.h"
#include "../../render/vk/shaders/utils.hlsl"

[[vk::push_constant]]
BatchDrawPushConst pushConst;

// Instances and views
[[vk::binding(0, 0)]]
StructuredBuffer<PackedViewData> viewDataBuffer;

[[vk::binding(1, 0)]]
StructuredBuffer<PackedInstanceData> engineInstanceBuffer;

[[vk::binding(2, 0)]]
StructuredBuffer<uint32_t> instanceOffsets;

// Draw information
[[vk::binding(0, 1)]]
RWStructuredBuffer<uint32_t> drawCount;

[[vk::binding(1, 1)]]
RWStructuredBuffer<DrawCmd> drawCommandBuffer;

[[vk::binding(2, 1)]]
RWStructuredBuffer<DrawDataBR> drawDataBuffer;

// Asset descriptor bindings
[[vk::binding(0, 2)]]
StructuredBuffer<PackedVertex> vertexDataBuffer;

[[vk::binding(1, 2)]]
StructuredBuffer<MeshData> meshDataBuffer;

struct V2F {
    [[vk::location(0)]] uint instanceID : TEXCOORD0;
    [[vk::location(1)]] uint meshID : TEXCOORD3;
};

float4 composeQuats(float4 a, float4 b)
{
    return float4(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

float3 rotateVec(float4 q, float3 v)
{
    float3 pure = q.xyz;
    float scalar = q.w;
    
    float3 pure_x_v = cross(pure, v);
    float3 pure_x_pure_x_v = cross(pure, pure_x_v);
    
    return v + 2.f * ((pure_x_v * scalar) + pure_x_pure_x_v);
}

PerspectiveCameraData unpackViewData(PackedViewData packed)
{
    const float4 d0 = packed.data[0];
    const float4 d1 = packed.data[1];
    const float4 d2 = packed.data[2];

    PerspectiveCameraData cam;
    cam.pos = d0.xyz;
    cam.rot = float4(d1.xyz, d0.w);
    cam.xScale = d1.w;
    cam.yScale = d2.x;
    cam.zNear = d2.y;

    return cam;
}

Vertex unpackVertex(PackedVertex packed)
{
    const float4 d0 = packed.data[0];
    const float4 d1 = packed.data[1];

    uint3 packed_normal_tangent = uint3(
        asuint(d0.w), asuint(d1.x), asuint(d1.y));

    float3 normal;
    float4 tangent_and_sign;
    decodeNormalTangent(packed_normal_tangent, normal, tangent_and_sign);

    Vertex vert;
    vert.position = float3(d0.x, d0.y, d0.z);
    vert.normal = normal;
    vert.tangentAndSign = tangent_and_sign;
    vert.uv = float2(d1.z, d1.w);

    return vert;
}

EngineInstanceData unpackEngineInstanceData(PackedInstanceData packed)
{
    const float4 d0 = packed.data[0];
    const float4 d1 = packed.data[1];
    const float4 d2 = packed.data[2];

    EngineInstanceData o;
    o.position = d0.xyz;
    o.rotation = float4(d1.xyz, d0.w);
    o.scale = float3(d1.w, d2.xy);
    o.objectID = asint(d2.z);

    return o;
}

#if 1
float3x3 toMat(float4 r)
{
    float x2 = r.x * r.x;
    float y2 = r.y * r.y;
    float z2 = r.z * r.z;
    float xz = r.x * r.z;
    float xy = r.x * r.y;
    float yz = r.y * r.z;
    float wx = r.w * r.x;
    float wy = r.w * r.y;
    float wz = r.w * r.z;

    return float3x3(
        float3(
            1.f - 2.f * (y2 + z2),
            2.f * (xy - wz),
            2.f * (xz + wy)),
        float3(
            2.f * (xy + wz),
            1.f - 2.f * (x2 + z2),
            2.f * (yz - wx)),
        float3(
            2.f * (xz - wy),
            2.f * (yz + wx),
            1.f - 2.f * (x2 + y2)));
}
#endif

void computeCompositeTransform(float3 obj_t,
                               float4 obj_r,
                               float3 cam_t,
                               float4 cam_r_inv,
                               out float3 to_view_translation,
                               out float4 to_view_rotation)
{
    to_view_translation = rotateVec(cam_r_inv, obj_t - cam_t);
    to_view_rotation = normalize(composeQuats(cam_r_inv, obj_r));
}

[shader("vertex")]
float4 vert(in uint vid : SV_VertexID,
            in uint draw_id : SV_InstanceID,
            out int layer_id : SV_RenderTargetArrayIndex,
            out int viewport_id : SV_ViewportArrayIndex,
            out V2F v2f) : SV_Position
{
    DrawDataBR draw_data = drawDataBuffer[draw_id];

    layer_id = draw_data.localViewID / pushConst.viewsPerLayer;
    viewport_id = draw_data.localViewID % pushConst.viewsPerLayer;

    Vertex vert = unpackVertex(vertexDataBuffer[vid]);
    uint instance_id = draw_data.instanceID;

    PerspectiveCameraData view_data =
        unpackViewData(viewDataBuffer[draw_data.viewID]);

    EngineInstanceData instance_data = unpackEngineInstanceData(
        engineInstanceBuffer[instance_id]);

    float3 to_view_translation;
    float4 to_view_rotation;
    computeCompositeTransform(instance_data.position, instance_data.rotation,
        view_data.pos, view_data.rot,
        to_view_translation, to_view_rotation);

    float3 view_pos =
        rotateVec(to_view_rotation, instance_data.scale * vert.position) +
            to_view_translation;

    float4 clip_pos = float4(
        view_data.xScale * view_pos.x,
        view_data.yScale * view_pos.z,
        view_data.zNear,
        view_pos.y);

    v2f.instanceID = draw_data.instanceID +
                     min(0, instanceOffsets[0]) +
                     min(0, drawCount[0]) +
                     min(0, drawCommandBuffer[0].vertexOffset) +
                     min(0, int(ceil(meshDataBuffer[0].vertexOffset)));

    v2f.meshID = draw_data.meshID;

    return clip_pos;
}

struct PixelOutput {
    uint2 ids : SV_Target0;
};

// We are basically packing 3 uints into 2. 21 bits per uint except for 22 
// for the instance ID
uint2 packVizBufferData(uint primitive_id, uint mesh_id, uint instance_id)
{
    primitive_id += 1;
    mesh_id += 1;
    instance_id += 1;

    uint d0 = primitive_id << 11;
    d0 |= 0x7FF & (instance_id >> 11);
    uint d1 = mesh_id << 11;
    d1 |= 0x7FF & instance_id;
    return uint2(d0, d1);
}

uint3 unpackVizBufferData(in uint2 data)
{
    uint primitive_id = data.x >> 11;
    uint mesh_id = data.y >> 11;
    uint instance_id = ((data.x & 0x7FF) << 11) | (data.y & 0x7FF);

    return uint3(primitive_id-1, mesh_id-1, instance_id-1);
}

[shader("pixel")]
PixelOutput frag(in V2F v2f,
                 in uint prim_id : SV_PrimitiveID)
{
    PixelOutput output;

    // IDs needs to hold the following information:
    // vertex_offset | index_start | instance_id | texture_id

    // Maximum value of texture_id is 100 (7 bits required)
    // instance_id will take up 25 bits
    // vertex_offset and index_start will both be 16 bits each
    output.ids = packVizBufferData(prim_id, v2f.meshID, v2f.instanceID);

    return output;
}
