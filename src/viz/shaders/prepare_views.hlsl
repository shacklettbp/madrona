#include "shader_common.h"
#include "../../render/vk/shaders/utils.hlsl"

[[vk::push_constant]]
PrepareViewPushConstant pushConst;

// Contains a sorted buffer of cameras by world ID. This compute shader will
// dispatch a workgroup for each view and have the 32 threads of the workgroup 
// process each instance data for that view (and perform culling).
[[vk::binding(0, 0)]]
StructuredBuffer<PerspectiveCameraDataBR > cameraBuffer;

[[vk::binding(1, 0)]]
StructuredBuffer<InstanceDataBR> instanceData;

[[vk::binding(2, 0)]]
StructuredBuffer<uint32_t> instanceOffsets;

[[vk::binding(0, 1)]]
RWStructuredBuffer<uint32_t> drawCount;

[[vk::binding(1, 1)]]
RWStructuredBuffer<DrawCmd> drawCommandBuffer;

[[vk::binding(2, 1)]]
RWStructuredBuffer<DrawData> drawDataBuffer;

// Asset descriptor bindings

[[vk::binding(0, 2)]]
StructuredBuffer<ObjectData> objectDataBuffer;

[[vk::binding(1, 2)]]
StructuredBuffer<MeshData> meshDataBuffer;

uint getNumInstancesForWorld(uint world_idx)
{
    if (world_idx == 0) {
        return instanceOffsets[0];
    } else if (world_idx == pushConst.numWorlds - 1) {
        return pushConst.numWorlds - instanceOffsets[world_idx-1];
    } else {
        return instanceOffsets[world_idx] - instanceOffsets[world_idx-1];
    }
}

uint getInstanceOffsetsForWorld(uint world_idx)
{
    if (world_idx == 0) {
        return 0;
    } else {
        return instanceOffsets[world_idx-1];
    }
}

struct SharedData {
    uint viewIdx;
    uint numInstancesPerThread;
    PerspectiveCameraDataBR packedCamera;
    uint offset;
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
        sm.packedCamera = cameraBuffer[sm.viewIdx];
        sm.offset = getInstanceOffsetsForWorld(sm.packedCamera.worldIDX);
        sm.numInstancesForWorld = getNumInstancesForWorld(sm.packedCamera.worldIDX);
        sm.numInstancesPerThread = sm.numInstancesForWorld /
                                           PREPARE_VIEW_WORKGROUP_SIZE;
    }

    GroupMemoryBarrierWithGroupSync();

    for (int i = 0; i < sm.numInstancesPerThread; ++i) {
        uint local_idx = i * sm.numInstancesPerThread;
        if (local_idx > sm.numInstancesForWorld)
            return;

        uint current_instance_idx = sm.offset +
                                    local_idx;

        InstanceDataBR instance_data = 
            instanceData[current_instance_idx];

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

            DrawData draw_data;
            draw_data.materialID = mesh.materialIndex;
            draw_data.instanceID =  current_instance_idx;

            drawCommandBuffer[draw_id] = draw_cmd;
            drawDataBuffer[draw_id] = draw_data;
        }

    }
}
