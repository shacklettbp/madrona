#include "shader_utils.hlsl"

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

[[vk::binding(0, 3)]]
StructuredBuffer<AABB> aabbs;

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

float4 quatInv(float4 q){
    return float4(-q.x,-q.y,-q.z,q.w)/length(q);
}

bool planeAABB(float4 plane, float3 pos, float3 extents){
    float r = dot(extents,float3(abs(plane.x),abs(plane.y),abs(plane.z)));
    float planeDist = dot(plane.xyz, pos) + plane.w;
    return planeDist + r >= 0;
}

struct SharedData {
    uint viewIdx;
    uint numInstancesPerThread;
    uint offset;
    PerspectiveCameraData camera;
    uint numInstancesForWorld;
    float4 nearPlane;
    float4 farPlane;
    float4 leftPlane;
    float4 rightPlane;
    float4 bottomPlane;
    float4 topPlane;
};

struct TransformedAABB {
    float3 pMin;
    float3 pMax;
};

float3x3 fromRS(float4 r, float3 s)
{
    float x2 = r.x * r.x;
    float y2 = r.y * r.y;
    float z2 = r.z * r.z;
    float xz = r.x * r.z;
    float xy = r.x * r.y;
    float yz = r.y * r.z;
    float wx = r.w * r.x;
    float wy = r.w * r.y;
    float wz = r.w * r.z;

    float3 ds = 2.f * s;

    return float3x3(
        s.x - ds.x * (y2 + z2),
        ds.x * (xy + wz),
        ds.x * (xz - wy),
        ds.y * (xy - wz),
        s.y - ds.y * (x2 + z2),
        ds.y * (yz + wx),
        ds.z * (xz + wy),
        ds.z * (yz - wx),
        s.z - ds.z * (x2 + y2)
    );
}

TransformedAABB applyTRS(in float3 pmin, in float3 pmax,
                         in EngineInstanceData instance_data)
{
    float3x3 rot_mat = fromRS(instance_data.rotation, instance_data.scale);

     // RTCD page 86
     TransformedAABB txfmed;
     [unroll] for (int i = 0; i < 3; i++) {
         txfmed.pMin[i] = txfmed.pMax[i] = instance_data.position[i];

         [unroll] for (int j = 0; j < 3; j++) {
             // Flipped because rot_mat is column major
             float e = rot_mat[j][i] * pmin[j];
             float f = rot_mat[j][i] * pmax[j];

             if (e < f) {
                 txfmed.pMin[i] += e;
                 txfmed.pMax[i] += f;
             } else {
                 txfmed.pMin[i] += f;
                 txfmed.pMax[i] += e;
             }
         }
     }

     return txfmed;
}

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
        PerspectiveCameraData view_data = unpackViewData(cameraBuffer[sm.viewIdx]);
        sm.camera = view_data;
        sm.offset = getInstanceOffsetsForWorld(sm.camera.worldID);
        sm.numInstancesForWorld = getNumInstancesForWorld(sm.camera.worldID);
        sm.numInstancesPerThread = (sm.numInstancesForWorld+31) / 32;

        // printf("Num instances %u for world\n", sm.numInstancesForWorld);

        float4 qInv = quatInv(view_data.rot);

        float3 front = rotateVec(qInv,float3(0, 1, 0));
        float3 up = rotateVec(qInv,float3(0, 0, 1));
        float3 right = cross(front,up);

        float zFar = 20000; //Dummy value for now
        float aspectRatio = view_data.yScale/view_data.xScale;
        float farPlaneHalfHeight = zFar; //Assumed fov of 90
        float farPlaneHalfWidth = farPlaneHalfHeight * aspectRatio;
        float3 farVec = zFar * front;

        sm.nearPlane = float4(front,dot(-front,view_data.pos+view_data.zNear*front));
        sm.farPlane = float4(-front,dot(front,view_data.pos+farVec));

        float3 leftNorm = cross(up,farVec - farPlaneHalfWidth*right);
        sm.leftPlane = float4(leftNorm,dot(-leftNorm,view_data.pos));
        float3 rightNorm = cross(farVec + farPlaneHalfWidth*right,up);
        sm.rightPlane = float4(rightNorm,dot(-rightNorm,view_data.pos));

        float3 bottomNorm = cross(right, farVec - up * farPlaneHalfHeight);
        sm.bottomPlane = float4(bottomNorm,dot(-bottomNorm,view_data.pos));
        float3 topNorm = cross(farVec + up * farPlaneHalfHeight, right);
        sm.topPlane = float4(topNorm,dot(-topNorm,view_data.pos));
    }

    GroupMemoryBarrierWithGroupSync();

    uint num_culled = 0;
    uint total = 0;

    for (int i = 0; i < sm.numInstancesPerThread; ++i) {
        uint local_idx = i * 32 + tid_local.x;
        if (local_idx >= sm.numInstancesForWorld)
            return;

        uint current_instance_idx = sm.offset +
                                    local_idx;

        EngineInstanceData instance_data = 
            unpackEngineInstanceData(instanceData[current_instance_idx]);

        AABB aabb = aabbs[instance_data.objectID];

        float3 pre_pmin = aabb.data[0].xyz;
        float3 pre_pmax = float3(aabb.data[0].w, aabb.data[1].xy);

        TransformedAABB txfm_aabb = applyTRS(pre_pmin, pre_pmax,
                                             instance_data);

        float3 center = (txfm_aabb.pMin + txfm_aabb.pMax)/2;
        float3 extents = txfm_aabb.pMax - center;

        // Don't do culling yet.

        int some_value = 0;

        total++;

        if((!planeAABB(sm.nearPlane,center,extents) || !planeAABB(sm.leftPlane,center,extents) ||
           !planeAABB(sm.rightPlane,center,extents) || !planeAABB(sm.bottomPlane,center,extents) ||
           !planeAABB(sm.topPlane,center,extents) || !planeAABB(sm.farPlane,center,extents))){

            num_culled++;

            some_value += (int)aabbs[0].data[0].x;

            continue;
        }

        ObjectData obj = objectDataBuffer[instance_data.objectID];

        uint draw_offset;
        InterlockedAdd(drawCount[gid.x], obj.numMeshes, draw_offset);

        for (int32_t i = 0; i < obj.numMeshes; i++) {
            MeshData mesh = meshDataBuffer[obj.meshOffset + i];

            uint draw_id = draw_offset + i;
            DrawCmd draw_cmd;
            draw_cmd.indexCount = mesh.numIndices;
            draw_cmd.instanceCount = 1 + min(abs(some_value), 0);
            draw_cmd.firstIndex = mesh.indexOffset;
            draw_cmd.vertexOffset = mesh.vertexOffset;
            draw_cmd.firstInstance = draw_id;

            DrawDataBR draw_data;
            draw_data.viewID = sm.viewIdx ;
            draw_data.instanceID =  current_instance_idx;
            draw_data.localViewID = gid.x;
            // This will allow us to access the vertex offset and the index offset
            draw_data.meshID = obj.meshOffset + i;

            drawCommandBuffer[gid.x * pushConst.maxDrawsPerView + draw_id] = draw_cmd;
            drawDataBuffer[gid.x * pushConst.maxDrawsPerView + draw_id] = draw_data;
        }
    }

    // printf("percent culled: %f\n", (float)num_culled / (float)total);

    GroupMemoryBarrierWithGroupSync();
}
