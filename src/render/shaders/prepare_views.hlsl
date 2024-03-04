#include "shader_common.h"
#include "../../render/vk/shaders/utils.hlsl"

[[vk::push_constant]]
PrepareViewPushConstant pushConst;

// Contains a sorted buffer of cameras by world ID. This compute shader will
// dispatch a workgroup for each view and have the 32 threads of the workgroup 
// process each instance data for that view (and perform culling).
[[vk::binding(0, 0)]]
StructuredBuffer<PackedViewData> cameraBuffer;

[[vk::binding(1, 0)]]
StructuredBuffer<PackedInstanceData> instanceData;

[[vk::binding(2, 0)]]
StructuredBuffer<uint32_t> instanceOffsets;

[[vk::binding(0, 1)]]
RWStructuredBuffer<uint32_t> drawCount;

[[vk::binding(1, 1)]]
RWStructuredBuffer<DrawCmd> drawCommandBuffer;

[[vk::binding(2, 1)]]
RWStructuredBuffer<DrawDataBR> drawDataBuffer;

// Asset descriptor bindings

[[vk::binding(0, 2)]]
StructuredBuffer<ObjectData> objectDataBuffer;

[[vk::binding(1, 2)]]
StructuredBuffer<MeshData> meshDataBuffer;

uint getNumInstancesForWorld(uint world_idx)
{
    if (world_idx == pushConst.numWorlds - 1) {
        return pushConst.numInstances - instanceOffsets[world_idx];
    } else {
        return instanceOffsets[world_idx+1] - instanceOffsets[world_idx];
    }
}

uint getInstanceOffsetsForWorld(uint world_idx)
{
    return instanceOffsets[world_idx];
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
    o.worldID = asint(d2.w);

    return o;
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
    cam.worldID = asint(d2.z);
    float pad = d2.w;

    return cam;
}

struct SharedData {
    uint viewIdx;
    uint numInstancesPerThread;
    uint offset;
    PerspectiveCameraData camera;
    uint numInstancesForWorld;
};

groupshared SharedData sm;

[numThreads(32, 1, 1)]
[shader("compute")]
void main(uint3 tid       : SV_DispatchThreadID,
          uint3 tid_local : SV_GroupThreadID,
          uint3 gid       : SV_GroupID)
{
    if (gid.x > pushConst.numViews)
        return;

    if (tid_local.x == 0) {
        // Each group processes a single view
        sm.viewIdx = gid.x + pushConst.offset;
        sm.camera = unpackViewData(cameraBuffer[sm.viewIdx]);
        sm.offset = getInstanceOffsetsForWorld(sm.camera.worldID);
        sm.numInstancesForWorld = getNumInstancesForWorld(sm.camera.worldID);
        sm.numInstancesPerThread = (sm.numInstancesForWorld+31) / 32;
    }

    GroupMemoryBarrierWithGroupSync();

    for (int i = 0; i < sm.numInstancesPerThread; ++i) {
        uint local_idx = i * 32 + tid_local.x;
        if (local_idx >= sm.numInstancesForWorld)
            return;

        uint current_instance_idx = sm.offset +
                                    local_idx;

        EngineInstanceData instance_data = 
            unpackEngineInstanceData(instanceData[current_instance_idx]);

        // Don't do culling yet.

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

            DrawDataBR draw_data;
            draw_data.viewID = sm.viewIdx ;
            draw_data.instanceID =  current_instance_idx;
            draw_data.layerID = gid.x;
            // This will allow us to access the vertex offset and the index offset
            draw_data.meshID = obj.meshOffset + i;

            drawCommandBuffer[draw_id] = draw_cmd;
            drawDataBuffer[draw_id] = draw_data;
        }
    }
}
