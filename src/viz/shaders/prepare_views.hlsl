#include "shader_common.h"
#include "utils.hlsl"

[[vk::push_constant]]
PrepareViewPushConstant pushConst;

// Contains a sorted buffer of cameras by world ID. This compute shader will
// dispatch a workgroup for each view and have the 32 threads of the workgroup 
// process each instance data for that view (and perform culling).
[[vk::binding(0, 0)]]
StructuredBuffer<PackedPerspectiveCameraData> cameraBuffer;

[[vk::binding(1, 0)]]
StructuredBuffer<WorldInstanceInfo> worldInstanceInfoBuffer;

[[vk::binding(2, 0)]]
StructuredBuffer<PackedInstanceData> instancesBuffer;

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


struct SharedData {
    uint viewIdx;
    uint numInstancesPerThread;
    PackedPerspectiveCameraData packedCamera;
    WorldInstanceInfo worldInfo;
};

groupshared SharedData sm;

[numThreads(32, 1, 1)]
[shader("compute")]
void main(uint3 tid       : SV_DispatchThreadID,
          uint3 tid_local : SV_GroupThreadID,
          uint3 gid       : SV_GroupID)
{
    if (gid.x > pushCont.numViews)
        return;

    if (tid_local == 0) {
        // Each group processes a single view
        sm.viewIdx = gid.x;
        sm.packedCamera = cameraBuffer[sharedData.viewIdx];
        sm.worldInfo = 
            worldInstanceInfoBuffer[sm.packedCamera.worldIDX];
        sm.numInstancesPerThread = sharedData.worldInfo.count /
                                           PREPARE_VIEW_WORKGROUP_SIZE;
    }

    GroupMemoryBarrierWithGroupSync();

    for (int i = 0; i < sm.numInstancesPerThread; ++i) {
        uint local_idx = i * sm.numInstancesPerThread;
        if (local_idx > sm.worldInfo.count)
            return;

        uint current_instance_idx = sm.worldInfo.offset +
                                    local_idx;

        PackedInstanceData instance_data = 
            instancesBuffer[current_instance_idx];

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
            draw_data.instanceID = instance_id;

            drawCommandBuffer[draw_id] = draw_cmd;
            drawDataBuffer[draw_id] = draw_data;
        }

    }
}
