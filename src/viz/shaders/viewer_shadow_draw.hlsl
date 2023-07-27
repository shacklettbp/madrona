#include "shader_common.h"
#include "../../render/vk/shaders/utils.hlsl"

[[vk::push_constant]]
DrawPushConst push_const;

[[vk::binding(0, 0)]]
StructuredBuffer<PackedViewData> viewDataBuffer;

[[vk::binding(1, 0)]]
StructuredBuffer<PackedInstanceData> engineInstanceBuffer;

[[vk::binding(2, 0)]]
StructuredBuffer<DrawData> drawDataBuffer;

[[vk::binding(3, 0)]]
StructuredBuffer<ShadowViewData> shadowViewDataBuffer;

// Asset descriptor bindings

[[vk::binding(0, 1)]]
StructuredBuffer<PackedVertex> vertexDataBuffer;

[[vk::binding(1, 1)]]
StructuredBuffer<MaterialData> materialBuffer;

#if 0
struct V2F {
    [[vk::location(0)]] float depth : TEXCOORD0;
};
#endif

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

[shader("vertex")]
float4 vert(in uint vid : SV_VertexID,
            in uint draw_id : SV_InstanceID) : SV_Position
{
    Vertex vert = unpackVertex(vertexDataBuffer[vid]);
    DrawData draw_data = drawDataBuffer[draw_id];
    float4 color = materialBuffer[draw_data.materialID].color;
    uint instance_id = draw_data.instanceID;

    float4x4 shadow_matrix = shadowViewDataBuffer[push_const.viewIdx].viewProjectionMatrix;

    EngineInstanceData instance_data = unpackEngineInstanceData(
        engineInstanceBuffer[instance_id]);

    float dummy = 0.00000000001f * float(drawDataBuffer[0].materialID + viewDataBuffer[0].data[0].w + materialBuffer[0].color.w);

    float4 world_space_pos = float4(
        instance_data.position + mul(toMat(instance_data.rotation), (instance_data.scale * vert.position)), 
        1.f + dummy);

    float4 clip_pos = mul(shadow_matrix, world_space_pos);

    return clip_pos;
}

[shader("pixel")]
float2 frag(in float4 position : SV_Position) : SV_Target0
{
    float depth = position.z;

    float dx = ddx(depth);
    float dy = ddy(depth);
    float sigma = depth * depth + 0.25 * (dx * dx + dy * dy);

    return float2(depth, sigma);
}
