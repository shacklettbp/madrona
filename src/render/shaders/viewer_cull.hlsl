#include "shader_utils.hlsl"

[[vk::push_constant]]
CullPushConst pushConst;

// Contains the view just for the fly cam
[[vk::binding(0, 0)]]
StructuredBuffer<PackedViewData> viewDataBuffer;

// Contains the instances for all the worlds
[[vk::binding(1, 0)]]
StructuredBuffer<PackedInstanceData> engineInstanceBuffer;

[[vk::binding(2, 0)]]
RWStructuredBuffer<uint32_t> drawCount;

[[vk::binding(3, 0)]]
RWStructuredBuffer<DrawCmd> drawCommandBuffer;

[[vk::binding(4, 0)]]
RWStructuredBuffer<DrawData> drawDataBuffer;

[[vk::binding(5, 0)]]
RWStructuredBuffer<int> instanceOffsets;

// Asset descriptor bindings

[[vk::binding(0, 1)]]
StructuredBuffer<ObjectData> objectDataBuffer;

[[vk::binding(1, 1)]]
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

struct SharedData {
    uint numInstances;
    uint numInstancesPerThread;
    uint instancesOffset;
};

groupshared SharedData sm;

// No actual culling performed yet
[numThreads(32, 1, 1)]
[shader("compute")]
void instanceCull(uint3 tid           : SV_DispatchThreadID,
                  uint3 tid_local     : SV_GroupThreadID,
                  uint3 gid           : SV_GroupID)
{
    if (tid_local.x == 0) {
        sm.numInstances = getNumInstancesForWorld(pushConst.worldIDX);
        sm.numInstancesPerThread = (sm.numInstances + pushConst.numThreads-1) /
                                   pushConst.numThreads;
        sm.instancesOffset = getInstanceOffsetsForWorld(pushConst.worldIDX);
        // printf("%d\n", sm.numInstances);
    }

    GroupMemoryBarrierWithGroupSync();

    for (int i = 0; i < sm.numInstancesPerThread; ++i) {
        uint local_idx = i * pushConst.numThreads + tid.x;

        if (local_idx >= sm.numInstances) {
            return;
        }

        uint current_instance_idx = sm.instancesOffset + local_idx;

        EngineInstanceData instance_data = unpackEngineInstanceData(
            engineInstanceBuffer[current_instance_idx]);

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
            
            if (instance_data.matID == -1) {
                draw_data.materialID = mesh.materialIndex;
            } else {
                draw_data.materialID = instance_data.matID;
            }

            draw_data.instanceID = current_instance_idx;
            draw_data.color = instance_data.color;

            drawCommandBuffer[draw_id] = draw_cmd;
            drawDataBuffer[draw_id] = draw_data;
        }
    }
}
