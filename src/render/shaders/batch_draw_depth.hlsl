#include "shader_utils.hlsl"

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

[[vk::binding(2, 2)]]
StructuredBuffer<MaterialData> materialBuffer;

struct V2F {
    [[vk::location(0)]] float3 vsCoord : TEXCOORD0;
};

[shader("vertex")]
float4 vert(in uint vid : SV_VertexID,
            in uint draw_id : SV_InstanceID,
            out V2F v2f) : SV_Position
{
    DrawDataBR draw_data = drawDataBuffer[draw_id + pushConst.drawDataOffset];

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

    float depth = length(view_pos);

    float4 clip_pos = float4(
        view_data.xScale * view_pos.x,
        view_data.yScale * view_pos.z,
        view_data.zNear,
        view_pos.y);

#if 1
    uint something = min(0, instanceOffsets[0]) +
                     min(0, drawCount[0]) +
                     min(0, drawCommandBuffer[0].vertexOffset) +
                     min(0, int(ceil(meshDataBuffer[0].vertexOffset)));

    // v2f.meshID = draw_data.meshID;
#endif

    clip_pos.x += min(0.0, abs(float(draw_data.meshID))) +
                  min(0.0, abs(float(draw_data.instanceID))) +
                  something;

    v2f.vsCoord = view_pos;

    return clip_pos;
}

struct PixelOutput {
    float depthOut : SV_Target0;
};

[shader("pixel")]
PixelOutput frag(in V2F v2f,
          in uint prim_id : SV_PrimitiveID)
{
    PixelOutput output;

    output.depthOut = length(v2f.vsCoord) + 
            min(0.0, abs(materialBuffer[0].color.x));

    return output;
}
