#include "shader_utils.hlsl"

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
StructuredBuffer<PackedLightData> lights;

// Atmosphere
[[vk::binding(4, 0)]]
Texture2D<float4> transmittanceLUT;

[[vk::binding(5, 0)]]
Texture2D<float4> irradianceLUT;

[[vk::binding(6, 0)]]
Texture3D<float4> scatteringLUT;

// Shadows
[[vk::binding(7, 0)]]
Texture2D<float2> shadowMap;

// Assume stuff is Y-UP from here
[[vk::binding(8, 0)]]
StructuredBuffer<ShadowViewData> shadowViewDataBuffer;

// Sampler
[[vk::binding(9, 0)]]
SamplerState linearSampler;

// Assume stuff from here is Y-UP
[[vk::binding(10, 0)]]
StructuredBuffer<SkyData> skyBuffer;

#include "lighting.h"

#define SHADOW_BIAS 0.002f

float linear_step(float low, float high, float v) {
    return clamp((v - low) / (high - low), 0, 1);
}

/* Shadowing is done using variance shadow mapping. */
float shadowFactorVSM(float3 world_pos, uint2 target_pixel)
{
    uint2 shadow_map_dim;
    shadowMap.GetDimensions(shadow_map_dim.x, shadow_map_dim.y);

    float2 texel_size = float2(1.f, 1.f) / float2(shadow_map_dim);

    float4 world_pos_v4 = float4(world_pos.xyz, 1.f);

    /* Light space position */
    float4 ls_pos = mul(shadowViewDataBuffer[pushConst.viewIdx].viewProjectionMatrix, 
                        world_pos_v4);

    ls_pos.xyz /= ls_pos.w;
    ls_pos.z += SHADOW_BIAS;

    /* UV to use when sampling in the shadow map. */
    float2 uv = ls_pos.xy * 0.5 + float2(0.5, 0.5);

    /* Only deal with points which are within the shadow map. */
    if (uv.x > 1.0 || uv.x < 0.0 || uv.y > 1.0 || uv.y < 0.0 ||
        ls_pos.z > 1.0 || ls_pos.z < 0.0)
        return 1.0;

    float2 moment = shadowMap.SampleLevel(linearSampler, uv, 0);

    float occlusion = 0.0f;

    float pcf_count = 1;

    for (int x = int(-pcf_count); x <= int(pcf_count); ++x) {
        for (int y = int(-pcf_count); y <= int(pcf_count); ++y) {
            float2 moment = shadowMap.SampleLevel(linearSampler, 
                                                  uv + float2(x, y) * texel_size, 0).rg;

            // Chebychev's inequality
            float p = (ls_pos.z > moment.x);
            float sigma = max(moment.y - moment.x * moment.x, 0.0);

            float dist_from_mean = (ls_pos.z - moment.x);

            float pmax = linear_step(0.9, 1.0, sigma / (sigma + dist_from_mean * dist_from_mean));
            float occ = min(1.0f, max(pmax, p));

            occlusion += occ;
        }
    }

    occlusion /= (pcf_count * 2.0f + 1.0f) * (pcf_count * 2.0f + 1.0f);

    return occlusion;
}

/* Coordinates which are prefixed with w are in world space. */
struct GBufferData {
    float3 wPosition;
    float3 wNormal;
    float3 albedo;
    float3 wCameraPos;
};

/* BRDF calculations. */
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

/* Gets the BRDF contribution on a pixel which lies on a surface for any generic
 * incoming radiance value. */
float3 directionalRadianceBRDF(in GBufferData gbuffer,
                               in float3 base_reflectivity,
                               in float roughness,
                               in float metalness,
                               in float3 view_direction,
                               in float3 incoming_radiance,
                               in float3 light_direction) {
    float3 view_dir = -view_direction;

    float3 halfway = normalize(light_direction + view_dir);

    float ndotv = max(dot(gbuffer.wNormal.xyz, view_dir), 0.000001f);
    float ndotl = max(dot(gbuffer.wNormal.xyz, light_direction), 0.000001f);
    float hdotv = max(dot(halfway, view_dir), 0.000001f);
    float ndoth = max(dot(gbuffer.wNormal.xyz, halfway), 0.000001f);

    float distribution_term = distributionGGX(ndoth, roughness);
    float smith_term = smithGGX(ndotv, ndotl, roughness);
    float3 fresnel_term = fresnel(hdotv, base_reflectivity);

    float3 specular = smith_term * distribution_term * fresnel_term;
    specular /= 4.0 * ndotv * ndotl;

    float3 kd = float3(1.0, 1.0, 1.0) - fresnel_term;

    kd *= 1.0f - metalness;

    return (kd * gbuffer.albedo.rgb / M_PI + specular) * incoming_radiance * ndotl;
}

/* Gets the radiance on a pixel which lies on a surface taking into account BRDF as
 * well as the sky transmittance and radiance incoming from the sun. Also takes into
 * account shadows using variance shadow maps. */
float3 accumulateSunRadianceBRDF(in GBufferData gbuffer, 
                                 float roughness, 
                                 float metal,
                                 float r,
                                 float mu_sun,
                                 uint2 target_pixel)
{ 
    float3 ret = float3(0.0, 0.0, 0.0);

    /* We get the radiance incoming from the sun to this point by taking into account
     * the transmittance of the sky. */
    float3 radiance_from_sun = skyBuffer[0].solarIrradiance.xyz * 
                               getTransmittanceToSun(skyBuffer[0], transmittanceLUT, r, mu_sun);

    ShaderLightData light = unpackLightData(lights[0]);
    ret += directionalRadianceBRDF(gbuffer,
                                   lerp(float3(0.04, 0.04, 0.04), gbuffer.albedo.rgb, metal),
                                   roughness,
                                   metal,
                                   normalize(gbuffer.wPosition.xyz - pushConst.viewPos.xyz),
                                   radiance_from_sun,
                                   normalize(-light.direction.xyz));

    float shadow_factor = shadowFactorVSM(gbuffer.wPosition, target_pixel);

    return ret * shadow_factor;
}

/* Caculates the radiance which actually arrives to the camera. This will require calling
 * accumulateSunRadianceBRDF to get the radiance hitting the surface, and then using
 * getSkyRadianceToPoint to calculate the amount of in/out scatter happening on the ray
 * from the surface point to the camera. */
float4 getPointRadianceBRDF(float roughness, float metal, in GBufferData gbuffer, uint2 target_pixel) 
{
    float3 sky_irradiance, sun_irradiance, point_radiance;
    ShaderLightData light = unpackLightData(lights[0]);

    { /* Calculate sun and sky irradiance which will contribute to the final BRDF. */
        float3 p = gbuffer.wPosition / 1000.0 - skyBuffer[0].wPlanetCenter.xyz;
        float3 normal = gbuffer.wNormal;

        float3 sun_direction = -normalize(light.direction.xyz);

        float3 view_direction = normalize(gbuffer.wPosition - pushConst.viewPos.xyz);

        float r = length(p);
        float mu_sun = dot(p, sun_direction) / r;

        sky_irradiance = getIrradiance(skyBuffer[0], irradianceLUT, r, mu_sun) *
                        (1.0 + dot(normal, p) / r) * 0.5;

        float3 accumulated_radiance = accumulateSunRadianceBRDF(gbuffer, roughness, metal, 
                                                                r, mu_sun, target_pixel);

        point_radiance = accumulated_radiance + gbuffer.albedo.rgb * (1.0 / M_PI) * sky_irradiance;
    }

    /* How much is scattered towards us. */
    float3 transmittance;
    float3 in_scatter = getSkyRadianceToPoint(skyBuffer[0], transmittanceLUT,
                                              scatteringLUT, scatteringLUT,
                                              pushConst.viewPos.xyz / 1000.0 - skyBuffer[0].wPlanetCenter.xyz,
                                              gbuffer.wPosition / 1000.0 - skyBuffer[0].wPlanetCenter.xyz, 0.0,
                                              -normalize(light.direction.xyz),
                                              transmittance);

    point_radiance = point_radiance * transmittance + in_scatter;

    return float4(point_radiance, 1.0);
}

/* Calculates the outgoing camera view ray for a given pixel. */
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

    float3 dir = screen.x * cam_right * right_scale - screen.y * cam_up * up_scale + cam_forward;

    return normalize(dir);
}

[numThreads(32, 32, 1)]
[shader("compute")]
void lighting(uint3 idx : SV_DispatchThreadID)
{
    uint2 target_dim;
    gbufferAlbedo.GetDimensions(target_dim.x, target_dim.y);

    if (idx.x < target_dim.x && idx.y < target_dim.y)
    {
        uint2 target_pixel = uint2(idx.x, idx.y);

        float3 outgoing_ray = getOutgoingRay((float2)target_pixel, (float2)target_dim);

        float4 color = gbufferAlbedo[target_pixel];

        float4 normal = gbufferNormal[target_pixel];
        float4 position = gbufferPosition[target_pixel];

        /* If normal.w is more than 0, this object was rasterized. */
        float point_alpha = normal.w;

        GBufferData gbuffer_data;
        gbuffer_data.wPosition = position.xyz;
        gbuffer_data.wNormal = normal.xyz;
        gbuffer_data.albedo = color.rgb;
        gbuffer_data.wCameraPos = pushConst.viewPos.xyz;

        float roughness = color.a;
        float metalness = position.a;

        /* Radiance from the rasterized pixel. */
        float4 point_radiance = getPointRadianceBRDF(roughness, metalness, 
                                                     gbuffer_data, target_pixel);

        ShaderLightData light = unpackLightData(lights[0]);
        float3 sun_direction = normalize(-light.direction.xyz);

        /* Incoming radiance from the sky: */
        float3 transmittance;
        float3 radiance = getSkyRadiance(skyBuffer[0], transmittanceLUT,
                                         scatteringLUT, scatteringLUT,
                                         (gbuffer_data.wCameraPos / 1000.0 - skyBuffer[0].wPlanetCenter.xyz),
                                         outgoing_ray, 0.0, sun_direction,
                                         transmittance);

        if (dot(outgoing_ray, sun_direction) >
                skyBuffer[0].sunSize.y * 0.99999) {
            radiance = radiance + transmittance * getSolarRadiance(skyBuffer[0]) * 0.06;
        }

        radiance = lerp(radiance, point_radiance.xyz, point_alpha);

        /* Tone Mapping. */
        float3 one = float3(1.0, 1.0, 1.0);
        float3 exp_value = exp(-radiance / float3(2.0f, 2.0f, 2.0f) * pushConst.exposure);

        float3 diff = one - exp_value;
        float3 out_color = diff;

        //float viz = abs(dot(normal.xyz, outgoing_ray));
        //out_color = float3(viz, viz, viz) * color.xyz + 1e-6f * out_color;

        gbufferAlbedo[target_pixel] = float4(out_color, 1.0);
    }
}
