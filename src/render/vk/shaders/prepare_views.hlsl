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

    }
}
