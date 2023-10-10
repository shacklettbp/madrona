#include "shader_common.h"
#include "../../render/vk/shaders/utils.hlsl"

// GBuffer descriptor bindings

[[vk::push_constant]]
DeferredLightingPushConstBR pushConst;

[[vk::binding(0, 0)]]
RWTexture2DArray<uint2> visBuffer;

[[vk::binding(1, 0)]]
RWTexture2DArray<float4> outputBuffer;

// Instances and views
[[vk::binding(0, 1)]]
StructuredBuffer<PackedViewData> viewDataBuffer;

[[vk::binding(1, 1)]]
StructuredBuffer<PackedInstanceData> engineInstanceBuffer;

[[vk::binding(2, 1)]]
StructuredBuffer<uint32_t> instanceOffsets;

// Asset descriptor bindings
[[vk::binding(0, 2)]]
StructuredBuffer<PackedVertex> vertexDataBuffer;

[[vk::binding(1, 2)]]
StructuredBuffer<MeshData> meshDataBuffer;

// Texture descriptor bindings
[[vk::binding(0, 3)]]
Texture2D<float4> materialTexturesArray[];

[[vk::binding(1, 3)]]
SamplerState linearSampler;

#include "lighting.h"

#define SHADOW_BIAS 0.002f

Vertex unpackVertex(PackedVertex packed)
{
    const float4 d0 = packed.data[0];
    const float4 d1 = packed.data[1];

    uint3 packed_normal_tangent = uint3(
        asuint(d0.w), asuint(d1.x), asuint(d1.y));

    float3 normal;
    float4 tangent_and_sign;
    decodeNormalTangent(packed_normal_tangent, normal, tangent_and_sign);

    Vertex vert;
    vert.position = float3(d0.x, d0.y, d0.z);
    vert.normal = normal;
    vert.tangentAndSign = tangent_and_sign;
    vert.uv = float2(d1.z, d1.w);

    return vert;
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

    return o;
}

uint hash(uint x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

float4 intToColor(uint inCol){
    if(inCol==0xffffffff) return float4(1,1,1,1);

    float a=((inCol & 0xff000000) >>24);
    float r=((inCol & 0xff0000) >> 16);
    float g=((inCol & 0xff00) >> 8);
    float b=((inCol & 0xff));
    return float4(r,g,b,255.0)/255.0;
}


[numThreads(1,32,32)]
[shader("compute")]
void lighting(uint3 gid : SV_GroupID,uint3 idx : SV_DispatchThreadID)
{
    uint3 target_dim;
    visBuffer.GetDimensions(target_dim.x, target_dim.y, target_dim.z);

    if (gid.x > pushConst.numViews)
            return;

    /*if (idx.y == 224 && idx.z == 224) {
        printf("Dims, %d,%d,%d,%d,%d,%d,(%d)\n", target_dim.x,target_dim.y,target_dim.z,idx.x,idx.y,idx.z,gid.x);
    }*/

    if (idx.y < target_dim.x && idx.z < target_dim.y)
    {
        uint3 target_pixel = uint3(idx.y, idx.z, gid.x);
        uint2 ids = visBuffer[target_pixel];
        uint triangleID = ids.x;
        uint instanceID = ids.y;

        EngineInstanceData instanceData = unpackEngineInstanceData(engineInstanceBuffer[instanceID]);
        Vertex vert = unpackVertex(vertexDataBuffer[triangleID]);
        ObjectData obj = objectDataBuffer[instanceData.objectID];

        float4 out_color = float4(min(0,abs(vert.normal.x)),0,0,0) + float4(min(0,abs(instanceData.scale.x)),0,0,0)
         + float4(min(0,abs(materialTexturesArray[0].SampleLevel(
                       linearSampler, float2(0,0), 0).x)),0,0,0) +
         intToColor(hash(triangleID));
        //out_color = out_color * min(0,ids.x) + float4(idx.y/200.0,idx.z/200.0,1,1);
        //out_color = max(out_color,float4(1,1,1,1));
        outputBuffer[target_pixel] = out_color;
    }
}
