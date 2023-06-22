#include "shader_common.h"
#include "../../render/vk/shaders/utils.hlsl"

// GBuffer descriptor bindings

[[vk::push_constant]]
DeferredLightingPushConst pushConst;

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
        float4 normal = gbufferNormal[targetPixel];
        float4 position = gbufferPosition[targetPixel];

        normal.xyz = normalize(normal.xyz);

        float4 light_dir = lights[0].lightDir;
        light_dir.xyz = normalize(light_dir.xyz);

        float diffuse = clamp(dot(-light_dir.xyz, normal.xyz), 0, 1);
        color.xyz += diffuse * lights[0].color.xyz;

        float3 reflected_light = reflect(light_dir.xyz, normal.xyz);
        float3 eye_vector = pushConst.viewPos.xyz - position.xyz;

        float specular = clamp(dot(reflected_light, normalize(eye_vector)), 0, 1);
        specular = pow(specular, 15.0);

        color.xyz += specular * lights[0].color.xyz;

        gbufferAlbedo[targetPixel] = color;
    }
}
