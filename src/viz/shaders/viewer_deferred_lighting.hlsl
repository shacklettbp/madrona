#include "shader_common.h"
#include "../../render/vk/shaders/utils.hlsl"

// GBuffer descriptor bindings

[[vk::binding(0, 0)]]
RWTexture2D<float4> gbufferAlbedo;

[[vk::binding(1, 0)]]
RWTexture2D<float4> gbufferNormal;

[[vk::binding(2, 0)]]
RWTexture2D<float4> gbufferPosition;

[[vk::binding(3, 0)]]
StructuredBuffer<DirectionalLight> lights;

[numThreads(32, 32, 1)]
[shader("compute")]
void lighting(uint3 idx : SV_DispatchThreadID)
{
    uint2 targetDim;
    gbufferAlbedo.GetDimensions(targetDim.x, targetDim.y);

    if (idx.x < targetDim.x && idx.y < targetDim.y)
    {
        uint2 targetPixel = uint2(idx.x, idx.y);

        float4 color = gbufferAlbedo[targetPixel];
        gbufferAlbedo[targetPixel] = color / 2.f;
    }
}
