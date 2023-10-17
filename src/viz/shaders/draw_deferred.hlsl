#include "shader_common.h"
#include "../../render/vk/shaders/utils.hlsl"

// GBuffer descriptor bindings

[[vk::push_constant]]
DeferredLightingPushConstBR pushConst;

// This is an array of all the textures
[[vk::binding(0, 0)]]
RWTexture2DArray<uint2> vizBuffer[];

[[vk::binding(1, 0)]]
RWTexture2DArray<float4> outputBuffer[];

#if 0
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
#endif

// #include "lighting.h"

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

uint hash(uint x)
{
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

float4 intToColor(uint inCol)
{
    if(inCol == 0xffffffff) return float4(1,1,1,1);

    float a = ((inCol & 0xff000000) >>24);
    float r = ((inCol & 0xff0000) >> 16);
    float g = ((inCol & 0xff00) >> 8);
    float b = ((inCol & 0xff));
    return float4(r,g,b,255.0)/255.0;
}


// idx.x is the x coordinate of the image
// idx.y is the y coordinate of the image
// idx.z is the global view index
[numThreads(32, 32, 1)]
[shader("compute")]
void lighting(uint3 idx : SV_DispatchThreadID)
{
    uint3 target_dim;
    vizBuffer[pushConst.imageIndex].GetDimensions(target_dim.x, target_dim.y, target_dim.z);

    if (idx.x >= target_dim.x || idx.y >= target_dim.y ||
        idx.z >= pushConst.totalNumViews) {
        return;
    }

    uint layer_idx = idx.z;
    uint3 target_pixel = uint3(idx.x, idx.y, layer_idx);

    uint2 ids = vizBuffer[pushConst.imageIndex][target_pixel];

    uint triangle_id = ids.x;
    uint instance_id = ids.y;

#if 0
    EngineInstanceData instanceData = unpackEngineInstanceData(engineInstanceBuffer[instance_id]);
    Vertex vert = unpackVertex(vertexDataBuffer[triangle_id]);

    uint zero_dummy = min(asint(viewDataBuffer[0].data[2].w), 0) +
                      min(asint(instanceData.worldID), 0) +
                      instanceOffsets[0] +
                      min(uint(abs(vert.normal.x)), 0) +
                      min(meshDataBuffer[0].vertexOffset, 0) +
                      min(0, abs(materialTexturesArray[0].SampleLevel(
                          linearSampler, float2(0,0), 0).x));
#endif

    uint h = hash(triangle_id) + hash(instance_id);
    float4 out_color = intToColor(h);

    outputBuffer[pushConst.imageIndex][target_pixel] = out_color;
}
