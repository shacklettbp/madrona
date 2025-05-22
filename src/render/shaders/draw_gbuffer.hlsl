#include "shader_utils.hlsl"

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

// Texture descriptor bindings
[[vk::binding(0, 2)]]
Texture2D<float4> materialTexturesArray[];

[[vk::binding(1, 2)]]
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

[shader("vertex")]
float4 vert(in uint vid : SV_VertexID,
            in uint draw_id : SV_InstanceID,
            out V2F v2f) : SV_Position
{
    DrawData draw_data = drawDataBuffer[draw_id];

    Vertex vert = unpackVertex(vertexDataBuffer[vid]);
    float4 color = materialBuffer[vert.materialIdx].color;
    uint instance_id = draw_data.instanceID;

    PerspectiveCameraData view_data =
        unpackViewData(viewDataBuffer[push_const.viewIdx]);

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

    // v2f.viewPos = view_pos;
#if 0
    v2f.normal = normalize(
        rotateVec(to_view_rotation, (vert.normal / instance_data.scale)));
#endif
    v2f.normal = normalize(
        rotateVec(instance_data.rotation, (vert.normal / instance_data.scale)));
    v2f.uv = vert.uv;
    v2f.color = color;
    v2f.position = rotateVec(instance_data.rotation,
                             instance_data.scale * vert.position) + instance_data.position;
    v2f.dummy = shadowViewDataBuffer[0].viewProjectionMatrix[0][0];
    v2f.texIdx = materialBuffer[vert.materialIdx].textureIdx;
    v2f.roughness = materialBuffer[vert.materialIdx].roughness;
    v2f.metalness = materialBuffer[vert.materialIdx].metalness;

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

    if (v2f.texIdx != -1) {
        output.color *= materialTexturesArray[v2f.texIdx].SampleLevel(
            linearSampler, float2(v2f.uv.x, 1.f - v2f.uv.y), 0);
    }

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
