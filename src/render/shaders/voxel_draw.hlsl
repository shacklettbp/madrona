#include "shader_utils.hlsl"

[[vk::push_constant]]
DrawPushConst push_const;

[[vk::binding(0, 0)]]
StructuredBuffer<PackedViewData> flycamBuffer;

// Asset descriptor bindings
[[vk::binding(1, 0)]]
StructuredBuffer<PackedVertex> vertexDataBuffer;

[[vk::binding(2, 0)]]
StructuredBuffer<PackedViewData> viewDataBuffer;

[[vk::binding(3, 0)]]
StructuredBuffer<int> viewOffsetsBuffer;

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

[shader("vertex")]
float4 vert(in uint vid : SV_VertexID,
            in uint draw_id : SV_InstanceID,
            out V2F v2f) : SV_Position
{
    Vertex vert = unpackVertex(vertexDataBuffer[vid]);
    float4 color = float4(1,1,1,1);

    PerspectiveCameraData view_data = getCameraData();
        //unpackViewData(viewDataBuffer[push_const.viewIdx]);

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
    //output.color = max(float4(1,1,1,1),output.color);
    //output.color = output.color*float4((output.normal.xyz+float3(1,1,1))*0.5,1);

    return output;
}