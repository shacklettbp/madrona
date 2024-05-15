#include "shader_common.h"
#include "../../render/vk/shaders/utils.hlsl"

// GBuffer descriptor bindings

[[vk::push_constant]]
DeferredLightingPushConstBR pushConst;

// This is an array of all the textures
[[vk::binding(0, 0)]]
RWTexture2DArray<uint2> vizBuffer[];

[[vk::binding(1, 0)]]
RWStructuredBuffer<uint32_t> rgbOutputBuffer;

[[vk::binding(2, 0)]]
RWStructuredBuffer<float> depthOutputBuffer;

[[vk::binding(0, 1)]]
StructuredBuffer<uint> indexBuffer;

// Instances and views
[[vk::binding(0, 2)]]
StructuredBuffer<PackedViewData> viewDataBuffer;

[[vk::binding(1, 2)]]
StructuredBuffer<PackedInstanceData> engineInstanceBuffer;

[[vk::binding(2, 2)]]
StructuredBuffer<uint32_t> instanceOffsets;

// Asset descriptor bindings
[[vk::binding(0, 3)]]
StructuredBuffer<PackedVertex> vertexDataBuffer;

[[vk::binding(1, 3)]]
StructuredBuffer<MeshData> meshDataBuffer;

[[vk::binding(2, 3)]]
StructuredBuffer<MaterialData> materialBuffer;

// Texture descriptor bindings
[[vk::binding(0, 4)]]
Texture2D<float4> materialTexturesArray[];

[[vk::binding(1, 4)]]
SamplerState linearSampler;


// Lighting
[[vk::binding(0, 5)]]
StructuredBuffer<DirectionalLight> lights;

[[vk::binding(1, 5)]]
Texture2D<float4> transmittanceLUT;

[[vk::binding(2, 5)]]
Texture2D<float4> irradianceLUT;

[[vk::binding(3, 5)]]
Texture3D<float4> scatteringLUT;

[[vk::binding(4, 5)]]
StructuredBuffer<SkyData> skyBuffer;


#include "lighting.h"

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
    vert.uv = unpackHalf2x16(asuint(d1.z));
    vert.materialIdx = asuint(d1.w);

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

PerspectiveCameraData unpackViewData(PackedViewData packed)
{
    const float4 d0 = packed.data[0];
    const float4 d1 = packed.data[1];
    const float4 d2 = packed.data[2];

    PerspectiveCameraData cam;
    cam.pos = d0.xyz;
    cam.rot = float4(d1.xyz, d0.w);
    cam.xScale = d1.w;
    cam.yScale = d2.x;
    cam.zNear = d2.y;

    return cam;
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

struct VertexData {
    float4 postMvp;
    float3 pos;
    float3 normal;
    float3 col;
    float2 uv;
};

float3 rotateVec(float4 q, float3 v)
{
    float3 pure = q.xyz;
    float scalar = q.w;
    
    float3 pure_x_v = cross(pure, v);
    float3 pure_x_pure_x_v = cross(pure, pure_x_v);
    
    return v + 2.f * ((pure_x_v * scalar) + pure_x_pure_x_v);
}

float4 composeQuats(float4 a, float4 b)
{
    return float4(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
}

void computeCompositeTransform(float3 obj_t,
                               float4 obj_r,
                               float3 cam_t,
                               float4 cam_r_inv,
                               out float3 to_view_translation,
                               out float4 to_view_rotation)
{
    to_view_translation = rotateVec(cam_r_inv, obj_t - cam_t);
    to_view_rotation = normalize(composeQuats(cam_r_inv, obj_r));
}

struct BarycentricDeriv
{
    float3 m_lambda;
    float3 m_ddx;
    float3 m_ddy;
};

BarycentricDeriv calcFullBary(float4 pt0, float4 pt1, float4 pt2, float2 pixelNdc, float2 winSize)
{
    BarycentricDeriv ret = (BarycentricDeriv)0;

    float3 invW = 1.0f / (float3(pt0.w, pt1.w, pt2.w));

    float2 ndc0 = pt0.xy * invW.x;
    float2 ndc1 = pt1.xy * invW.y;
    float2 ndc2 = pt2.xy * invW.z;

    float invDet = rcp(determinant(float2x2(ndc2 - ndc1, ndc0 - ndc1)));
    ret.m_ddx = float3(ndc1.y - ndc2.y, ndc2.y - ndc0.y, ndc0.y - ndc1.y) * invDet * invW;
    ret.m_ddy = float3(ndc2.x - ndc1.x, ndc0.x - ndc2.x, ndc1.x - ndc0.x) * invDet * invW;
    float ddxSum = dot(ret.m_ddx, float3(1,1,1));
    float ddySum = dot(ret.m_ddy, float3(1,1,1));

    float2 deltaVec = pixelNdc - ndc0;
    float interpInvW = invW.x + deltaVec.x*ddxSum + deltaVec.y*ddySum;
    float interpW = rcp(interpInvW);

    ret.m_lambda.x = interpW * (invW[0] + deltaVec.x*ret.m_ddx.x + deltaVec.y*ret.m_ddy.x);
    ret.m_lambda.y = interpW * (0.0f    + deltaVec.x*ret.m_ddx.y + deltaVec.y*ret.m_ddy.y);
    ret.m_lambda.z = interpW * (0.0f    + deltaVec.x*ret.m_ddx.z + deltaVec.y*ret.m_ddy.z);

    ret.m_ddx *= (2.0f/winSize.x);
    ret.m_ddy *= (2.0f/winSize.y);
    ddxSum    *= (2.0f/winSize.x);
    ddySum    *= (2.0f/winSize.y);

    ret.m_ddy *= -1.0f;
    ddySum    *= -1.0f;

    float interpW_ddx = 1.0f / (interpInvW + ddxSum);
    float interpW_ddy = 1.0f / (interpInvW + ddySum);

    ret.m_ddx = interpW_ddx*(ret.m_lambda*interpInvW + ret.m_ddx) - ret.m_lambda;
    ret.m_ddy = interpW_ddy*(ret.m_lambda*interpInvW + ret.m_ddy) - ret.m_lambda;  

    return ret;
}

float3 interpolateWithDeriv(BarycentricDeriv deriv, float v0, float v1, float v2)
{
    float3 mergedV = float3(v0, v1, v2);
    float3 ret;
    ret.x = dot(mergedV, deriv.m_lambda);
    ret.y = dot(mergedV, deriv.m_ddx);
    ret.z = dot(mergedV, deriv.m_ddy);
    return ret;
}

float interpolateFloat(in BarycentricDeriv deriv, in float v0, in float v1, in float v2)
{
    float3 mergedV = float3(v0, v1, v2);
    return dot(mergedV, deriv.m_lambda);
}

float3 interpolateVec3(in BarycentricDeriv deriv, in float3 v0, in float3 v1, in float3 v2)
{
    return v0 * deriv.m_lambda.x +
           v1 * deriv.m_lambda.y +
           v2 * deriv.m_lambda.z;
}

float2 interpolateVec2(in BarycentricDeriv deriv, in float2 v0, in float2 v1, in float2 v2)
{
    return v0 * deriv.m_lambda.x +
           v1 * deriv.m_lambda.y +
           v2 * deriv.m_lambda.z;
}

struct UVInterpolation {
    float2 interp;
    float2 dx;
    float2 dy;
};

UVInterpolation interpolateUVs(
    BarycentricDeriv deriv,
    in float2 uv0,
    in float2 uv1,
    in float2 uv2)
{
    float3 attr0 = float3(uv0.x, uv1.x, uv2.x);
    float3 attr1 = float3(uv0.y, uv1.y, uv2.y);

    UVInterpolation result;

    result.interp.x = interpolateFloat(deriv, attr0.x, attr0.y, attr0.z);
    result.interp.y = interpolateFloat(deriv, attr1.x, attr1.y, attr1.z);

    result.dx.x = dot(attr0, deriv.m_ddx);
    result.dx.y = dot(attr1, deriv.m_ddx);
    result.dy.x = dot(attr0, deriv.m_ddy);
    result.dy.y = dot(attr1, deriv.m_ddy);
    return result;
}

uint zeroDummy()
{
    uint zero_dummy = min(asuint(viewDataBuffer[0].data[2].w), 0) +
                      min(asuint(engineInstanceBuffer[0].data[0].x), 0) +
                      min(asuint(vertexDataBuffer[0].data[0].x), 0) +
                      min(meshDataBuffer[0].vertexOffset, 0) +
                      min(indexBuffer[0], 0) +
                      min(0, abs(materialTexturesArray[0].SampleLevel(
                          linearSampler, float2(0,0), 0).x)) +
                      min(instanceOffsets[0], 0) +
                      min(0.0, abs(transmittanceLUT.SampleLevel(
                          linearSampler, float2(0.0, 0.0f), 0).x)) +
                      min(0.0, abs(irradianceLUT.SampleLevel(
                          linearSampler, float2(0.0, 0.0f), 0).x)) +
                      min(0.0, abs(scatteringLUT.SampleLevel(
                          linearSampler, float3(0.0, 0.0f, 0.0f), 0).x)) +
                      min(0.0, abs(skyBuffer[0].solarIrradiance.x)) +
                      min(0.0, abs(lights[0].color.x));


    return zero_dummy;
}

float3 toWorldSpace(in EngineInstanceData instance_data,
                    in Vertex vert)
{
    return rotateVec(instance_data.rotation, instance_data.scale * vert.position) +
           instance_data.position;
}

float3 toViewSpace(in EngineInstanceData instance_data,
                   in PerspectiveCameraData view_data,
                   in Vertex vert)
{
    float3 to_view_translation;
    float4 to_view_rotation;
    computeCompositeTransform(instance_data.position, instance_data.rotation,
            view_data.pos, view_data.rot,
            to_view_translation, to_view_rotation);

    float3 view_pos =
        rotateVec(to_view_rotation, instance_data.scale * vert.position) +
        to_view_translation;

    return view_pos;
}

float4 toNDC(in PerspectiveCameraData view_data,
             in float3 view_pos)
{
    float4 clip_pos = float4(
        view_data.xScale * view_pos.x,
        view_data.yScale * view_pos.z,
        view_data.zNear,
        view_pos.y);

    return clip_pos;
}

// We are basically packing 3 uints into 2. 21 bits per uint except for 22 
// for the instance ID
uint2 packVizBufferData(uint primitive_id, uint mesh_id, uint instance_id)
{
    primitive_id += 1;
    mesh_id += 1;
    instance_id += 1;

    uint d0 = primitive_id << 11;
    d0 |= 0x7FF & (instance_id >> 11);
    uint d1 = mesh_id << 11;
    d1 |= 0x7FF & instance_id;
    return uint2(d0, d1);
}

uint3 unpackVizBufferData(in uint2 data)
{
    uint primitive_id = data.x >> 11;
    uint mesh_id = data.y >> 11;
    uint instance_id = ((data.x & 0x7FF) << 11) | (data.y & 0x7FF);

    return uint3(primitive_id-1, mesh_id-1, instance_id-1);
}

struct GBufferData {
    float3 wPosition;
    float3 wNormal;
    float3 albedo;
    float3 wCameraPos;
};

/* BRDF calculations. */
float distributionGGX(float ndoth, float roughness) 
{
    float a2 = roughness * roughness;
    float f = (ndoth * a2 - ndoth) * ndoth + 1.0f;
    return a2 / (M_PI * f * f);
}

float smithGGX(float ndotv, float ndotl, float roughness) 
{
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    float ggx1 = ndotv / (ndotv * (1.0f - k) + k);
    float ggx2 = ndotl / (ndotl * (1.0f - k) + k);
    return ggx1 * ggx2;
}

float3 fresnel(float hdotv, float3 base) 
{
    return base + (1.0f - base) * pow(1.0f - clamp(hdotv, 0.0f, 1.0f), 5.0f);
}

float3 fresnelRoughness(float ndotv, float3 base, float roughness) 
{
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
                               in float3 light_direction) 
{
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
                                 in PerspectiveCameraData camera_data,
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

    ret += directionalRadianceBRDF(gbuffer,
                                   lerp(float3(0.04, 0.04, 0.04), gbuffer.albedo.rgb, metal),
                                   roughness,
                                   metal,
                                   normalize(gbuffer.wPosition.xyz - camera_data.pos.xyz),
                                   radiance_from_sun,
                                   normalize(-lights[0].lightDir.xyz));

    return ret * 1.0;
}

/* Caculates the radiance which actually arrives to the camera. This will require calling
 * accumulateSunRadianceBRDF to get the radiance hitting the surface, and then using
 * getSkyRadianceToPoint to calculate the amount of in/out scatter happening on the ray
 * from the surface point to the camera. */
float4 getPointRadianceBRDF(float roughness, float metal, 
                            in GBufferData gbuffer, in PerspectiveCameraData camera_data,
                            uint2 target_pixel) 
{
    float3 sky_irradiance, sun_irradiance, point_radiance;

    { /* Calculate sun and sky irradiance which will contribute to the final BRDF. */
        float3 p = gbuffer.wPosition / 1000.0 - skyBuffer[0].wPlanetCenter.xyz;
        float3 normal = gbuffer.wNormal;

        float3 sun_direction = -normalize(lights[0].lightDir.xyz);

        float3 view_direction = normalize(gbuffer.wPosition - camera_data.pos.xyz);

        float r = length(p);
        float mu_sun = dot(p, sun_direction) / r;

        sky_irradiance = getIrradiance(skyBuffer[0], irradianceLUT, r, mu_sun) *
                        (1.0 + dot(normal, p) / r) * 0.5;

        float3 accumulated_radiance = accumulateSunRadianceBRDF(gbuffer, camera_data, roughness, metal, 
                                                                r, mu_sun, target_pixel);

        point_radiance = accumulated_radiance + gbuffer.albedo.rgb * (1.0 / M_PI) * sky_irradiance;
    }

    /* How much is scattered towards us. */
    float3 transmittance;
    float3 in_scatter = getSkyRadianceToPoint(skyBuffer[0], transmittanceLUT,
                                              scatteringLUT, scatteringLUT,
                                              camera_data.pos.xyz / 1000.0 - skyBuffer[0].wPlanetCenter.xyz,
                                              gbuffer.wPosition / 1000.0 - skyBuffer[0].wPlanetCenter.xyz, 0.0,
                                              -normalize(lights[0].lightDir.xyz),
                                              transmittance);

    point_radiance = point_radiance * transmittance + in_scatter;

    return float4(point_radiance, 1.0);
}

/* Calculates the outgoing camera view ray for a given pixel. */
float3 getOutgoingRay(float2 target_pixel, float2 target_dim, in PerspectiveCameraData camera)
{
    float3 cam_forward = normalize(rotateVec(camera.rot, float3(0, 1, 0)));
    float3 cam_up = normalize(rotateVec(camera.rot, float3(0, 0, 1)));
    float3 cam_right = normalize(rotateVec(camera.rot, float3(1, 0, 0)));

    float aspect_ratio = target_dim.x / target_dim.y;
    float tan_fov = -1.0 / camera.yScale;

    float right_scale = aspect_ratio * tan_fov;
    float up_scale = tan_fov;

    float2 raster = float2(target_pixel.x + 0.5, target_pixel.y + 0.5);
    float2 screen = float2((2.0f * raster.x) / target_dim.x - 1.0f,
                           (2.0f * raster.y) / target_dim.y - 1.0f);

    float3 dir = screen.x * cam_right * right_scale - screen.y * cam_up * up_scale + cam_forward;

    return normalize(dir);
}

float linearToSRGB(float v)
{
    if (v <= 0.00031308f) {
        return 12.92f * v;
    } else {
        return 1.055f*pow(v,(1.f / 2.4f)) - 0.055f;
    }
}

uint32_t linearToSRGB8(float3 rgb)
{
    float3 srgb = float3(
        linearToSRGB(rgb.x), 
        linearToSRGB(rgb.y), 
        linearToSRGB(rgb.z));

    uint3 quant = (uint3)(255 * clamp(srgb, 0.f, 1.f));

    return quant.r | (quant.g << 8) | (quant.b << 16) | ((uint32_t)255 << 24);
}

// idx.x is the x coordinate of the image
// idx.y is the y coordinate of the image
// idx.z is the global view index
[numThreads(32, 32, 1)]
[shader("compute")]
void lighting(uint3 idx : SV_DispatchThreadID)
{
    uint view_idx = idx.z;

    uint num_views_per_image = pushConst.maxImagesXPerTarget * 
                               pushConst.maxImagesYPerTarget;

    // Figure out which image to render to
    uint target_idx = view_idx / num_views_per_image;

    // View index within that target
    uint target_view_idx = view_idx % num_views_per_image;

    uint target_view_idx_x = target_view_idx %
                             pushConst.maxImagesXPerTarget;
    uint target_view_idx_y = target_view_idx /
                             pushConst.maxImagesXPerTarget;

    float x_pixel_offset = target_view_idx_x * pushConst.viewDim;
    float y_pixel_offset = target_view_idx_y * pushConst.viewDim;

    if (idx.x >= pushConst.viewDim || idx.y >= pushConst.viewDim) {
        return;
    }

    uint3 vbuffer_pixel = uint3(idx.x, idx.y, 0);

    float2 vbuffer_pixel_clip =
        float2(float(vbuffer_pixel.x) + 0.5f, float(vbuffer_pixel.y) + 0.5f) /
        float2(pushConst.viewDim, pushConst.viewDim);

    vbuffer_pixel_clip = vbuffer_pixel_clip * 2.0f - float2(1.0f, 1.0f);
    vbuffer_pixel_clip.y *= -1.0;

    // Apply the offset when reading the pixel value from the image
    uint2 data = vizBuffer[target_idx][vbuffer_pixel + 
                 uint3(x_pixel_offset, y_pixel_offset, 0)];

    uint3 unpacked_viz_data = unpackVizBufferData(data);

    uint primitive_id = unpacked_viz_data.x;
    uint mesh_id = unpacked_viz_data.y;
    uint instance_id = unpacked_viz_data.z;

    GBufferData gbuffer_data;
    float roughness;
    float metalness;

    bool was_rasterized = false;

    PerspectiveCameraData view_data = unpackViewData(viewDataBuffer[view_idx]);

    if (primitive_id != 0xFFFFFFFF || mesh_id != 0xFFFFFFFF || instance_id != 0xFFFFFFFF) {
        was_rasterized = true;

        MeshData mesh_data = meshDataBuffer[mesh_id];

        uint32_t material_idx = 0;

        uint index_start = mesh_data.indexOffset + primitive_id * 3;

        EngineInstanceData instance_data = unpackEngineInstanceData(engineInstanceBuffer[instance_id]);

        // This is the interpolated vertex information at the pixel that the given thread is on
        VertexData vertices[3];

        float3 w_inv;
        float3 w_values;

        for (int i = 0; i < 3; ++i) {
            uint vertex_idx = indexBuffer[index_start + i];
            PackedVertex packed = vertexDataBuffer[mesh_data.vertexOffset + vertex_idx];
            Vertex vert = unpackVertex(packed);

            if (i == 0) {
                material_idx = vert.materialIdx;
            }

            float3 to_view_translation;
            float4 to_view_rotation;
            computeCompositeTransform(instance_data.position, instance_data.rotation,
                    view_data.pos, view_data.rot,
                    to_view_translation, to_view_rotation);

            float3 view_pos =
                rotateVec(to_view_rotation, instance_data.scale * vert.position) +
                to_view_translation;

            float3 world_pos =
                rotateVec(instance_data.rotation, instance_data.scale * vert.position) +
                instance_data.position;

            float4 clip_pos = float4(
                    view_data.xScale * view_pos.x,
                    -view_data.yScale * view_pos.z,
                    view_data.zNear,
                    view_pos.y);

            w_values[i] = clip_pos.w;
            w_inv[i] = 1.0f / clip_pos.w;

            vertices[i].postMvp = clip_pos;
            vertices[i].pos = world_pos;

            vertices[i].normal = normalize(
                    rotateVec(instance_data.rotation, (vert.normal / instance_data.scale)));

            vertices[i].uv = float2(vert.uv.x, 1.0 - vert.uv.y);
        }

        // Get barrycentric coordinates
        BarycentricDeriv bc = calcFullBary(vertices[0].postMvp,
                vertices[1].postMvp,
                vertices[2].postMvp,
                vbuffer_pixel_clip,
                float2(pushConst.viewDim, pushConst.viewDim));

        float interpolated_w = 1.0f / dot(w_inv, bc.m_lambda);

        // Get interpolated normal
        float3 normal = normalize(interpolateVec3(bc, vertices[0].normal, vertices[1].normal, vertices[2].normal));

        // Get interpolated position
        float3 position = interpolateVec3(bc, vertices[0].pos, vertices[1].pos, vertices[2].pos);

        // Get interpolated UVs
        UVInterpolation uv = interpolateUVs(bc, vertices[0].uv, vertices[1].uv, vertices[2].uv);

        MaterialData material_data = materialBuffer[mesh_data.materialIndex];

        float4 color = material_data.color;
        if (material_data.textureIdx != -1) {
            color *= materialTexturesArray[material_data.textureIdx].SampleLevel(
                    linearSampler, uv.interp, 0);
        }

        gbuffer_data.wPosition = position;
        gbuffer_data.wNormal = normal;
        gbuffer_data.albedo = color.rgb;
        gbuffer_data.wCameraPos = view_data.pos;
        roughness = material_data.roughness;
        metalness = material_data.metalness;
    }

#if 0
    const float exposure = 20.0f;

    // Lighting calculations
    float3 outgoing_ray = getOutgoingRay(
        float2(vbuffer_pixel.xy),
        float2(pushConst.viewDim, pushConst.viewDim),
        view_data);
    float4 point_radiance = getPointRadianceBRDF(roughness, metalness, gbuffer_data, view_data, vbuffer_pixel.xy);

    float3 sun_direction = normalize(-lights[0].lightDir.xyz);

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

    if (was_rasterized) {
        radiance = point_radiance.xyz;
    }

    // float point_alpha = was_rasterized ? 1.0 : 0.0;
    // radiance = lerp(radiance, point_radiance.xyz, point_alpha);

    /* Tone Mapping. */
    float3 one = float3(1.0, 1.0, 1.0);
    float3 exp_value = exp(-radiance / float3(2.0f, 2.0f, 2.0f) * exposure);

    float3 diff = one - exp_value;
    float3 out_color = diff;
#else
    // float3 out_color = gbuffer_data.wNormal.xyz;
    float3 out_color;
    if (was_rasterized) {
        out_color = gbuffer_data.albedo.xyz;
    } else {
        out_color = float3(0, 0, 0);
    }
#endif

    out_color += zeroDummy();

    uint32_t out_pixel_idx =
        view_idx * pushConst.viewDim * pushConst.viewDim +
        idx.y * pushConst.viewDim + idx.x;

    rgbOutputBuffer[out_pixel_idx] = linearToSRGB8(out_color); 

    float depth;
    if (was_rasterized) {
        depth = length(gbuffer_data.wPosition - gbuffer_data.wCameraPos);
    } else {
        depth = 0.f;
    }
    depthOutputBuffer[out_pixel_idx] = depth;
}
