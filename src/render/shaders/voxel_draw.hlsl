#include "shader_common.h"
#include "../../render/vk/shaders/utils.hlsl"

[[vk::push_constant]]
DrawPushConst push_const;

[[vk::binding(0, 0)]]
StructuredBuffer<PackedViewData> viewDataBuffer;

// Asset descriptor bindings
[[vk::binding(1, 0)]]
StructuredBuffer<PackedVertex> vertexDataBuffer;

// Texture descriptor bindings
[[vk::binding(0, 1)]]
Texture2D<float4> materialTexturesArray[];

[[vk::binding(1, 1)]]
SamplerState linearSampler;

struct V2F {
    [[vk::location(0)]] float3 normal : TEXCOORD0;
    [[vk::location(1)]] float3 position : TEXCOORD1;
    [[vk::location(2)]] float4 color : TEXCOORD2;
    [[vk::location(3)]] float dummy : TEXCOORD3;
    [[vk::location(4)]] float2 uv : TEXCOORD4;
    [[vk::location(5)]] int texIdx : TEXCOORD5;
    [[vk::location(6)]] float roughness : TEXCOORD6;
    [[vk::location(7)]] float metalness : TEXCOORD7;
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
    vert.normal = float3(d1.y,d1.z,d1.w);
    vert.tangentAndSign = tangent_and_sign;
    vert.uv = float2(d0.w, d1.x);

    return vert;
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
            out V2F v2f) : SV_Position
{
    Vertex vert = unpackVertex(vertexDataBuffer[vid]);
    float4 color = float4(1,1,1,1);

    PerspectiveCameraData view_data =
        unpackViewData(viewDataBuffer[push_const.viewIdx]);

    float3 to_view_translation;
    float4 to_view_rotation;

    float3 objectScale = float3(1,1,1);
    float3 objectPos = float3(0,0,0);
    float4 objectRotation = float4(0,0,0,1);

    computeCompositeTransform(objectPos, objectRotation,
        view_data.pos, view_data.rot,
        to_view_translation, to_view_rotation);

    float3 view_pos =
        rotateVec(to_view_rotation, objectScale * vert.position) +
            to_view_translation;

    float4 clip_pos = float4(
        view_data.xScale * view_pos.x,
        view_data.yScale * view_pos.z,
        view_data.zNear,
        view_pos.y);

    // v2f.viewPos = view_pos;
#if 0
    v2f.normal = normalize(
        rotateVec(to_view_rotation, (vert.normal / objectScale)));
#endif
    v2f.normal = normalize(
        rotateVec(objectRotation, (vert.normal / objectScale)));
    v2f.uv = vert.uv;
    v2f.color = color;
    v2f.position = rotateVec(objectRotation,
                             objectScale * vert.position) + objectPos;
    v2f.dummy = 1;
    v2f.texIdx = 2;
    v2f.roughness = 0;
    v2f.metalness = 0;

    return clip_pos;
}

struct PixelOutput {
    float4 color : SV_Target0;
    float4 normal : SV_Target1;
    float4 position : SV_Target2;
};

[shader("pixel")]
PixelOutput frag(in V2F v2f)
{
    PixelOutput output;
    output.color = v2f.color;
    output.color.a = v2f.roughness;
    output.normal = float4(normalize(v2f.normal), 1.f);
    output.position = float4(v2f.position, v2f.dummy * 0.0000001f);
    output.position.a += v2f.metalness;

    // output.color.rgb = v2f.normal.xyz;

    if ( v2f.texIdx != -1) {
        output.color *= materialTexturesArray[v2f.texIdx].SampleLevel(
            linearSampler, float2(v2f.uv.x, 1.f - v2f.uv.y), 0);
    }
    //output.color = max(float4(1,1,1,1),output.color);
    //output.color = output.color*float4((output.normal.xyz+float3(1,1,1))*0.5,1);

    return output;
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
