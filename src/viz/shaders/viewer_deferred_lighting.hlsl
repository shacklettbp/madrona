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

// Atmosphere
[[vk::binding(4, 0)]]
RWTexture2D<float4> transmittance;

[[vk::binding(5, 0)]]
RWTexture2D<float4> irradiance;

[[vk::binding(6, 0)]]
RWTexture3D<float4> mie;

[[vk::binding(7, 0)]]
RWTexture3D<float4> scattering;

// Shadows
[[vk::binding(8, 0)]]
Texture2D<float> shadowMap;

[[vk::binding(9, 0)]]
StructuredBuffer<ShadowViewData> shadowViewDataBuffer;

// Sampler
[[vk::binding(10, 0)]]
SamplerState linearSampler;

float shadowFactor(float3 world_pos)
{
    float4 world_pos_v4 = float4(world_pos.x, world_pos.z, world_pos.y, 1.f);

    // Light space position
    float4 ls_pos = mul(shadowViewDataBuffer[0].viewProjectionMatrix, world_pos_v4);
    ls_pos.xyz /= ls_pos.w;

    float2 uv = ls_pos.xy * 0.5 + float2(0.5, 0.5);

    float map_depth = shadowMap.SampleLevel(linearSampler, uv, 0);

    if (map_depth > ls_pos.z)
        return 0.1f; // In shadow
    else
        return 1.0f;
}

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

        float4 trans = transmittance[uint2(0.f, 0.f)];
        float4 irr = irradiance[uint2(0.f, 0.f)];
        float4 mieSa = mie[uint3(0.f, 0.f, 0.f)];
        float4 scat = scattering[uint3(0.f, 0.f, 0.f)];

        normal.xyz = normalize(normal.xyz);

        float shadow_factor = shadowFactor(position.xyz);

        float4 light_dir = lights[0].lightDir;
        light_dir.xyz = normalize(light_dir.xyz);

        float diffuse = clamp(dot(-light_dir.xyz, normal.xyz), 0, 1);
        color.xyz += diffuse * lights[0].color.xyz;

        float3 reflected_light = reflect(light_dir.xyz, normal.xyz);
        float3 eye_vector = pushConst.viewPos.xyz - position.xyz;

        float specular = clamp(dot(reflected_light, normalize(eye_vector)), 0, 1);
        specular = pow(specular, 15.0);

        color.xyz += specular * lights[0].color.xyz;

        color *= shadow_factor;

        gbufferAlbedo[targetPixel] = color + (trans + irr + mieSa + scat) * 0.00001f;
    }
}
