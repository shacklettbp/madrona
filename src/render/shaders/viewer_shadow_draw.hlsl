#include "shader_utils.hlsl"

[[vk::push_constant]]
DrawPushConst push_const;

[[vk::binding(0, 0)]]
StructuredBuffer<PackedViewData> flycamBuffer;

[[vk::binding(1, 0)]]
StructuredBuffer<PackedInstanceData> engineInstanceBuffer;

[[vk::binding(2, 0)]]
StructuredBuffer<DrawData> drawDataBuffer;

[[vk::binding(3, 0)]]
StructuredBuffer<ShadowViewData> shadowViewDataBuffer;

[[vk::binding(4, 0)]]
StructuredBuffer<PackedViewData> viewDataBuffer;

[[vk::binding(5, 0)]]
StructuredBuffer<int> viewOffsetsBuffer;

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


[shader("vertex")]
float4 vert(in uint vid : SV_VertexID,
            in uint draw_id : SV_InstanceID) : SV_Position
{
    Vertex vert = unpackVertex(vertexDataBuffer[vid]);
    DrawData draw_data = drawDataBuffer[draw_id];
    float4 color = materialBuffer[vert.materialIdx].color;
    uint instance_id = draw_data.instanceID;

    float4x4 shadow_matrix = shadowViewDataBuffer[push_const.viewIdx].viewProjectionMatrix;

    EngineInstanceData instance_data = unpackEngineInstanceData(
        engineInstanceBuffer[instance_id]);

    float dummy = 0.00000000001f * float(drawDataBuffer[0].materialID + 
        flycamBuffer[0].data[0].w + materialBuffer[0].color.w +
        viewDataBuffer[0].data[0].w + float(viewOffsetsBuffer[0]));

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
