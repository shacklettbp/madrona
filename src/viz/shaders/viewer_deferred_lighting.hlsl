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

// Assume stuff is Y-UP from here
[[vk::binding(3, 0)]]
StructuredBuffer<DirectionalLight> lights;

// Atmosphere
[[vk::binding(4, 0)]]
Texture2D<float4> transmittanceLUT;

[[vk::binding(5, 0)]]
Texture2D<float4> irradianceLUT;

[[vk::binding(6, 0)]]
Texture3D<float4> mieLUT;

[[vk::binding(7, 0)]]
Texture3D<float4> scatteringLUT;

// Shadows
[[vk::binding(8, 0)]]
Texture2D<float> shadowMap;

// Assume stuff is Y-UP from here
[[vk::binding(9, 0)]]
StructuredBuffer<ShadowViewData> shadowViewDataBuffer;

// Sampler
[[vk::binding(10, 0)]]
SamplerState linearSampler;

// Assume stuff from here is Y-UP
[[vk::binding(11, 0)]]
StructuredBuffer<SkyData> skyBuffer;

[[vk::binding(12, 0)]]
Texture3D<float4> shadowOffsetsLUT;

#include "lighting.h"

#define SHADOW_PCF_PIXEL_TOLERANCE 3

#define  SHADOW_OFFSET_OUTER  32
#define  SHADOW_OFFSET_FILTER_SIZE  8
#define SHADOW_MAP_RANDOM_RADIUS 5

float4 get_offset_at(uint3 offset_coord, uint3 offset_dim)
{
    float3 uv = (float3(offset_coord) + float3(0.5, 0.5, 0.5)) / float3(offset_dim);
    return shadowOffsetsLUT.SampleLevel(linearSampler, uv, 0);
}

float shadowFactorRandomSample(float3 world_pos, uint2 target_pixel)
{
    uint2 shadow_map_dim;
    shadowMap.GetDimensions(shadow_map_dim.x, shadow_map_dim.y);

    uint3 shadow_offset_dim;
    shadowOffsetsLUT.GetDimensions(shadow_offset_dim.x, shadow_offset_dim.y, shadow_offset_dim.z);


    float4 world_pos_v4 = float4(world_pos.xyz, 1.f);

    // Light space position
    float4 ls_pos = mul(shadowViewDataBuffer[pushConst.viewIdx].viewProjectionMatrix, world_pos_v4);
    ls_pos.xyz /= ls_pos.w;
    ls_pos.z += 0.001f;

    float2 uv = ls_pos.xy * 0.5 + float2(0.5, 0.5);

    if (uv.x > 1.0 || uv.x < 0.0 || uv.y > 1.0 || uv.y < 0.0 ||
        ls_pos.z > 1.0 || ls_pos.z < -1.0)
        return 1.0;



    uint2 sample_slice = //mod(target_pixel, uint2(SHADOW_OFFSET_OUTER, SHADOW_OFFSET_OUTER));
        uint2(target_pixel.x % SHADOW_OFFSET_OUTER, target_pixel.y % SHADOW_OFFSET_OUTER);
    uint3 offset_coord = float3(0, sample_slice);


    float total = 0.0;

    float texel_width = 1.0 / (float)shadow_map_dim.x;
    float texel_height = 1.0 / (float)shadow_map_dim.y;
    float2 texel_size = float2(texel_width, texel_height);

    

    for (int i = 0; i < 4; ++i) {
        offset_coord.x = i;
        float4 offsets = /*shadowOffsetsLUT[offset_coord]*/
           get_offset_at(offset_coord, shadow_offset_dim) * (float)SHADOW_MAP_RANDOM_RADIUS;

        float2 sample_uv = uv + offsets.xy * texel_size;

        float map_depth = shadowMap.SampleLevel(linearSampler, sample_uv, 0);
        if (map_depth < ls_pos.z) {
           total += 1.0;
        } else {
           total += 0.0;
        }

        sample_uv = uv + offsets.zw * texel_size;
        map_depth = shadowMap.SampleLevel(linearSampler, sample_uv, 0);
        if (map_depth < ls_pos.z) {
           total += 1.0;
        } else {
           total += 0.0;
        }
    }

    float shadow_factor = total / 8.0;

    // If the shadow factor lies in between in/out of shadow, need to do more random sampling.
    if (shadow_factor != 0.0 && shadow_factor != 1.0) {
        int num_samples = int(SHADOW_OFFSET_FILTER_SIZE * SHADOW_OFFSET_FILTER_SIZE / 2);
        
        for (int i = 4; i < num_samples; ++i) {
            offset_coord.x = i;
            // float4 offsets = shadowOffsetsLUT[offset_coord] * (float)SHADOW_MAP_RANDOM_RADIUS;
            float4 offsets = /*shadowOffsetsLUT[offset_coord]*/
               get_offset_at(offset_coord, shadow_offset_dim) * (float)SHADOW_MAP_RANDOM_RADIUS;

            float2 sample_uv = uv + offsets.xy * texel_size;

            float map_depth = shadowMap.SampleLevel(linearSampler, sample_uv, 0);
            if (map_depth < ls_pos.z) {
                total += 1.0;
            } else {
                total += 0.0;
            }

            sample_uv = uv + offsets.zw * texel_size;
            map_depth = shadowMap.SampleLevel(linearSampler, sample_uv, 0);
            if (map_depth < ls_pos.z) {
                total += 1.0;
            } else {
                total += 0.0;
            }
        }

        shadow_factor = total / float(num_samples * 2);
    }

    return shadow_factor;
}

float shadowFactor(float3 world_pos, float3 world_normal)
{
    float4 world_pos_v4 = float4(world_pos.xyz, 1.f);

    // Light space position
    float4 ls_pos = mul(shadowViewDataBuffer[pushConst.viewIdx].viewProjectionMatrix, world_pos_v4);
    ls_pos.xyz /= ls_pos.w;

    float2 uv = ls_pos.xy * 0.5 + float2(0.5, 0.5);

    int extent = 1;


    uint2 shadow_map_dim;
    shadowMap.GetDimensions(shadow_map_dim.x, shadow_map_dim.y);


    float shadow = 0.0;
    float2 texel_size = 1.0 / float2(shadow_map_dim);
    for(int x = -extent; x <= extent; ++x) {
        for(int y = -extent; y <= extent; ++y) {
            float map_depth = shadowMap.SampleLevel(linearSampler, uv + float2(x, y) * texel_size, 0);
            shadow += (ls_pos.z > map_depth ? 1.0 : 0.0);
        }    
    }
    shadow /= float((extent * 2 + 1) * (extent * 2 + 1));

    return shadow;
}

// Assume that position and normal are in Y-up coordinate system TODO: Change to Z-UP
struct GBufferData {
    float3 wPosition;
    float3 wNormal;
    float3 albedo;
    float3 wCameraPos;
};

float distributionGGX(float ndoth, float roughness) {
    float a2 = roughness * roughness;
    float f = (ndoth * a2 - ndoth) * ndoth + 1.0f;
    return a2 / (M_PI * f * f);
}

float smithGGX(float ndotv, float ndotl, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    float ggx1 = ndotv / (ndotv * (1.0f - k) + k);
    float ggx2 = ndotl / (ndotl * (1.0f - k) + k);
    return ggx1 * ggx2;
}

float3 fresnel(float hdotv, float3 base) {
    return base + (1.0f - base) * pow(1.0f - clamp(hdotv, 0.0f, 1.0f), 5.0f);
}

float3 fresnelRoughness(float ndotv, float3 base, float roughness) {
    float one_minus_rough = 1.0f - roughness;
    return base + (max(float3(one_minus_rough, one_minus_rough, one_minus_rough), base) - base) *
        pow(1.0f - ndotv, 5.0f);
}

// Assume Y-UP for parameters
float3 directionalRadianceBRDF(
        in GBufferData gbuffer,
        in float3 baseReflectivity,
        in float roughness,
        in float metalness,
        in float3 viewDirection,
        in float3 incomingRadiance,
        in float3 lightDirection) {
    float3 viewDir = -viewDirection;

    float3 halfway = normalize(lightDirection + viewDir);

    float ndotv = max(dot(gbuffer.wNormal.xyz, viewDir), 0.000001f);
    float ndotl = max(dot(gbuffer.wNormal.xyz, lightDirection), 0.000001f);
    float hdotv = max(dot(halfway, viewDir), 0.000001f);
    float ndoth = max(dot(gbuffer.wNormal.xyz, halfway), 0.000001f);

    float distributionTerm = distributionGGX(ndoth, roughness);
    float smithTerm = smithGGX(ndotv, ndotl, roughness);
    float3 fresnelTerm = fresnel(hdotv, baseReflectivity);

    float3 specular = smithTerm * distributionTerm * fresnelTerm;
    specular /= 4.0 * ndotv * ndotl;

    float3 kd = float3(1.0, 1.0, 1.0) - fresnelTerm;

    kd *= 1.0f - metalness;

    return (kd * gbuffer.albedo.rgb / M_PI + specular) * incomingRadiance * ndotl;
}

// Assume Y-UP for parameters
float3 accumulateSunRadianceBRDF(in GBufferData gbuffer, float roughness, float metal, float r, float muSun, uint2 target_pixel)
{ 
    float3 ret = float3(0.0, 0.0, 0.0);

    ret += directionalRadianceBRDF(
            gbuffer,
            lerp(float3(0.04, 0.04, 0.04), gbuffer.albedo.rgb, metal),
            roughness,
            metal,
            normalize(gbuffer.wPosition.xyz - pushConst.viewPos.xzy),
            skyBuffer[0].solarIrradiance.xyz * getTransmittanceToSun(
                skyBuffer[0], transmittanceLUT, r, muSun),
                normalize(-lights[0].lightDir.xzy));

    // float shadow_factor = shadowFactor(gbuffer.wPosition, gbuffer.wNormal);
    float shadow_factor = shadowFactorRandomSample(gbuffer.wPosition, target_pixel);

    return ret * shadow_factor;
}

// Assume Y-UP for parameters
float4 getPointRadianceBRDF(float roughness, float metal, in GBufferData gbuffer, uint2 target_pixel) 
{
    float3 skyIrradiance, sunIrradiance, pointRadiance;
    { // Calculate sun and sky irradiance which will contribute to the final BRDF
        float3 p = gbuffer.wPosition / 1000.0 - skyBuffer[0].wPlanetCenter.xyz;
        float3 normal = gbuffer.wNormal;

        float3 sunDirection = -normalize(lights[0].lightDir.xzy);

        float3 viewDirection = normalize(gbuffer.wPosition - pushConst.viewPos.xzy);

        float r = length(p);
        float muSun = dot(p, sunDirection) / r;

        skyIrradiance = getIrradiance(skyBuffer[0], irradianceLUT, r, muSun) *
            (1.0 + dot(normal, p) / r) * 0.5;

        float3 accumulatedRadiance = accumulateSunRadianceBRDF(
                gbuffer, roughness, metal, r, muSun, target_pixel);

        pointRadiance = accumulatedRadiance + gbuffer.albedo.rgb * (1.0 / M_PI) * skyIrradiance;
    }

    /* How much is scattered towards us */
    float3 transmittance;
    float3 inScatter = getSkyRadianceToPoint(
            skyBuffer[0], transmittanceLUT,
            scatteringLUT, mieLUT,
            pushConst.viewPos.xzy / 1000.0 - skyBuffer[0].wPlanetCenter.xyz,
            gbuffer.wPosition / 1000.0 - skyBuffer[0].wPlanetCenter.xyz, 0.0,
            -normalize(lights[0].lightDir.xzy),
            transmittance);

    pointRadiance = pointRadiance * transmittance + inScatter;

    return float4(pointRadiance, 1.0);
}

// Y-UP
float3 getOutgoingRay(float2 target_pixel, float2 target_dim)
{
    float aspect_ratio = target_dim.x / target_dim.y;
    float tan_fov = tan(pushConst.fovy / 2.0f);

    float right_scale = aspect_ratio * tan_fov;
    float up_scale = tan_fov;

    float2 raster = float2(target_pixel.x + 0.5, target_pixel.y + 0.5);
    float2 screen = float2((2.0f * raster.x) / target_dim.x - 1.0f,
                           (2.0f * raster.y) / target_dim.y - 1.0f);

    float3 cam_right = shadowViewDataBuffer[pushConst.viewIdx].cameraRight.xyz;
    float3 cam_up = shadowViewDataBuffer[pushConst.viewIdx].cameraUp.xyz;
    float3 cam_forward = shadowViewDataBuffer[pushConst.viewIdx].cameraForward.xyz;

    float3 dir = -screen.x * cam_right * right_scale - screen.y * cam_up * up_scale + cam_forward;

    return normalize(dir);
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

        float3 outgoing_ray = getOutgoingRay((float2)targetPixel, (float2)targetDim);

        float4 color = gbufferAlbedo[targetPixel];

        // TODO: Make sure to convert this whole thing into Z-UP coordinate system
        float4 normal = gbufferNormal[targetPixel].xzyw;
        float4 position = gbufferPosition[targetPixel].xzyw;

        // If this is more than 0, this object was rasterized
        float point_alpha = normal.w;

        GBufferData gbuffer_data;
        gbuffer_data.wPosition = position.xyz;
        gbuffer_data.wNormal = normal.xyz;
        gbuffer_data.albedo = color.rgb;
        gbuffer_data.wCameraPos = pushConst.viewPos.xzy;

        float roughness = color.a;
        float metalness = position.a;

        // Radiance at the rasterized pixel
        float4 point_radiance = getPointRadianceBRDF(roughness, metalness, gbuffer_data, targetPixel);

        float3 sun_direction = normalize(-lights[0].lightDir.xzy);

        // Incoming radiance from the sky:
        float3 transmittance;
        float3 radiance = getSkyRadiance(
                skyBuffer[0], transmittanceLUT,
                scatteringLUT, mieLUT,
                (gbuffer_data.wCameraPos / 1000.0 - skyBuffer[0].wPlanetCenter.xyz),
                outgoing_ray, 0.0, sun_direction,
                transmittance);

        if (dot(outgoing_ray, sun_direction) >
                skyBuffer[0].sunSize.y * 0.99999) {
            radiance = radiance + transmittance * getSolarRadiance(skyBuffer[0]) * 0.06;
        }

        radiance = lerp(radiance, point_radiance.xyz, point_alpha);

        // Tone Mapping
        float3 one = float3(1.0, 1.0, 1.0);
        float3 expValue =
            exp(-radiance / float3(2.0f, 2.0f, 2.0f) * pushConst.exposure);

        float3 diff = one - expValue;
        float3 out_color = diff;

        gbufferAlbedo[targetPixel] = float4(out_color, 
            0.0000001f * shadowMap.SampleLevel(linearSampler, float2(0.0f, 0.0f), 0).x +
            0.0000001f * mieLUT.SampleLevel(linearSampler, float3(0.0f, 0.0f, 0.0f), 0).x);
    }
}
