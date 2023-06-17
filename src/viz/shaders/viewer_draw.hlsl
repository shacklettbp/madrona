#include "shader_common.h"
#include "../../render/vk/shaders/utils.hlsl"

[[vk::push_constant]]
DrawPushConst push_const;

[[vk::binding(0, 0)]]
cbuffer ViewData {
    PackedViewData viewData;
};

[[vk::binding(1, 0)]]
StructuredBuffer<PackedInstanceData> engineInstanceBuffer;

// Asset descriptor bindings

[[vk::binding(0, 1)]]
StructuredBuffer<PackedVertex> vertexDataBuffer;

struct VSOut {
    float4 clipPos : SV_POSITION;
    [[vk::location(0)]] float3 viewPos : TEXCOORD0;
    [[vk::location(1)]] float3 normal : TEXCOORD1;
    [[vk::location(2)]] float2 uv : TEXCOORD2;
};

struct PSIn {
    [[vk::location(0)]] float3 viewPos : TEXCOORD0;
    [[vk::location(1)]] float3 normal : TEXCOORD1;
    [[vk::location(2)]] float2 uv : TEXCOORD2;
};

float4 composeQuats(float4 a, float4 b)
{
    return float4(
        (a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z),
        (a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y),
        (a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x),
        (a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w));
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
    cam.position = d0.xyz;
    cam.rotation = float4(d1.xyz, d0.w);
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

    EngineInstanceData out;
    out.position = d0.xyz;
    out.rotation = float4(d1.xyz, d0.w);
    out.scale = float3(d1.w, d2.xy);
    out.objectID = asint(d2.z);

    return out;
}
 
void computeTransform(float3 obj_t,
                      float4 obj_r,
                      float3 cam_t,
                      float4 cam_r_inv,
                      out float3x3 to_view_rot,
                      out float3 to_view_translation)
{
    to_view_translation = rotateVec(cam_r_inv, obj_t - cam_t);

    float4 r = composeQuats(cam_r_inv, obj_r);
    r = normalize(r);

    float x2 = r.x * r.x;
    float y2 = r.y * r.y;
    float z2 = r.z * r.z;
    float xz = r.x * r.z;
    float xy = r.x * r.y;
    float yz = r.y * r.z;
    float wx = r.w * r.x;
    float wy = r.w * r.y;
    float wz = r.w * r.z;

    to_view_rot = float3x3(
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
VSOut vert(uint vid : SV_VertexID, uint instance_id : SV_InstanceID)
{
    PackedVertex packed_vert = vertexData[vid];
    Vertex vert = unpackVertex(packed_vert);

    PerspectiveCameraData view_data = 
        unpackViewData(viewDataBuffer[push_const.viewIdx]);

    EngineInstanceData instance_data = unpackEngineInstanceData(
        engineInstanceBuffer[instance_id]);

    float3x3 to_view_rot;
    float3 to_view_translation;
    computeTransform(instance_data.position, instance_data.rotation,
                     view_data.position, view_data.rotation,
                     to_view_rot, to_view_translation);

    float3 view_pos = to_view_rot * (obj_scale * vert.position) +
        to_view_translation;
    float4 clip_pos = float4(
        draw_data.projScale.x * view_pos.x,
        draw_data.projScale.y * view_pos.z,
        draw_data.projZNear,
        view_pos.y);

    VSOut out;
    out.clipPos = clip_pos;
    out.viewPos = view_pos;
    out.normal = normalize(to_view_rot * (vert.normal / obj_scale));
    out.uv = vert.uv;

    return out;
}

[shader("pixel")]
float4 frag(PSIn in) : SV_TARGET
{
    float hit_angle = max(dot(normalize(in.normal),
                              normalize(-in.viewPos)), 0.f);

    return float4(float4(hit_angle), 1.0);
}

#if 0
DrawInstanceData unpackDrawInstanceData(PackedDrawInstanceData data)
{
    const float4 d0 = data.packed[0];
    const float4 d1 = data.packed[1];
    const float4 d2 = data.packed[2];
    const float4 d3 = data.packed[3];
    const float4 d4 = data.packed[4];

    DrawInstanceData out;

    float3 rot_col0 = d0.xyz;
    float3 rot_col1 = float3(d0.w, d1.xy);
    float3 rot_col2 = float3(d1.zw, d2.x);

    out.toViewRot = float3x3(
        float3(rot_col0.x, rot_col1.x, rot_col2.x),
        float3(rot_col0.y, rot_col1.y, rot_col2.y),
        float3(rot_col0.z, rot_col1.z, rot_col2.z),
    );
    out.toViewTranslation = d2.yzw;
    out.objScale = d3.xyz;
    out.viewIdx = asint(d3.w);
    out.projScale = d4.xy;
    out.projZNear = d4.z;

    return out;
}
#endif
