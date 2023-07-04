#include "shader_common.h"
#include "../../render/vk/shaders/utils.hlsl"

[[vk::push_constant]]
CullPushConst push_const;

[[vk::binding(0, 0)]]
StructuredBuffer<PackedViewData> viewDataBuffer;

[[vk::binding(1, 0)]]
StructuredBuffer<PackedInstanceData> engineInstanceBuffer;

[[vk::binding(2, 0)]]
RWStructuredBuffer<uint32_t> drawCount;

[[vk::binding(3, 0)]]
RWStructuredBuffer<DrawCmd> drawCommandBuffer;

[[vk::binding(4, 0)]]
RWStructuredBuffer<DrawData> drawDataBuffer;

// Asset descriptor bindings

[[vk::binding(0, 1)]]
StructuredBuffer<ObjectData> objectDataBuffer;

[[vk::binding(1, 1)]]
StructuredBuffer<MeshData> meshDataBuffer;

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

// No actual culling performed yet
[numThreads(32, 1, 1)]
[shader("compute")]
void instanceCull(uint3 idx : SV_DispatchThreadID)
{
    uint32_t instance_id = idx.x;
    if (instance_id >= push_const.numInstances) {
        return;
    }

    EngineInstanceData instance_data = unpackEngineInstanceData(
        engineInstanceBuffer[instance_id]);

    ObjectData obj = objectDataBuffer[instance_data.objectID];

    uint draw_offset;
    InterlockedAdd(drawCount[0], obj.numMeshes, draw_offset);

    for (int32_t i = 0; i < obj.numMeshes; i++) {
        MeshData mesh = meshDataBuffer[obj.meshOffset + i];

        uint draw_id = draw_offset + i;
        DrawCmd draw_cmd;
        draw_cmd.indexCount = mesh.numIndices;
        draw_cmd.instanceCount = 1;
        draw_cmd.firstIndex = mesh.indexOffset;
        draw_cmd.vertexOffset = mesh.vertexOffset;
        draw_cmd.firstInstance = draw_id;

        DrawData draw_data;
        draw_data.materialID = mesh.materialIndex;
        draw_data.instanceID = instance_id;

        drawCommandBuffer[draw_id] = draw_cmd;
        drawDataBuffer[draw_id] = draw_data;
    }
}
