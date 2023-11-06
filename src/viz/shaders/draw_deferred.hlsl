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

[[vk::binding(0, 1)]]
StructuredBuffer<uint> indexBuffer;

#if 1
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
    uint zero_dummy = min(asint(viewDataBuffer[0].data[2].w), 0) +
                      min(asint(engineInstanceBuffer[0].data[0].x), 0) +
                      min(asuint(vertexDataBuffer[0].data[0].x), 0) +
                      min(meshDataBuffer[0].vertexOffset, 0) +
                      min(indexBuffer[0], 0) +
                      min(0, abs(materialTexturesArray[0].SampleLevel(
                          linearSampler, float2(0,0), 0).x)) +
                      min(instanceOffsets[0], 0);

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

    float2 target_pixel_clip = (float2(float(target_pixel.x) + 0.5f,
                                       float(target_pixel.y) + 0.5f) /
                                            float2(target_dim.x, target_dim.y));
    target_pixel_clip = target_pixel_clip * 2.0f - float2(1.0f, 1.0f);
    target_pixel_clip.y *= -1.0;

    uint2 data = vizBuffer[pushConst.imageIndex][target_pixel];

    uint3 unpacked_viz_data = unpackVizBufferData(data);

    uint primitive_id = unpacked_viz_data.x;
    uint mesh_id = unpacked_viz_data.y;
    uint instance_id = unpacked_viz_data.z;

    if (primitive_id == 0xFFFFFFFF && mesh_id == 0xFFFFFFFF && instance_id == 0xFFFFFFFF) {
        return;
    }

    MeshData mesh_data = meshDataBuffer[mesh_id];
    MaterialData material_data = materialBuffer[mesh_data.materialIndex];

    uint index_start = mesh_data.indexOffset + primitive_id * 3;
    uint view_idx = layer_idx + pushConst.maxLayersPerImage * pushConst.imageIndex;

    EngineInstanceData instance_data = unpackEngineInstanceData(engineInstanceBuffer[instance_id]);
    PerspectiveCameraData view_data = unpackViewData(viewDataBuffer[view_idx]);

    // This is the interpolated vertex information at the pixel that the given thread is on
    VertexData vertices[3];

    float3 w_inv;
    float3 w_values;

    for (int i = 0; i < 3; ++i) {
        uint vertex_idx = indexBuffer[index_start + i];
        PackedVertex packed = vertexDataBuffer[mesh_data.vertexOffset + vertex_idx];
        Vertex vert = unpackVertex(packed);

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

        vertices[i].uv = vert.uv;
    }

    // Get barrycentric coordinates
    BarycentricDeriv bc = calcFullBary(vertices[0].postMvp,
                                       vertices[1].postMvp,
                                       vertices[2].postMvp,
                                       target_pixel_clip,
                                       float2(target_dim.xy));

    float interpolated_w = 1.0f / dot(w_inv, bc.m_lambda);

    // Get interpolated normal
    float3 normal = normalize(interpolateVec3(bc, vertices[0].normal, vertices[1].normal, vertices[2].normal));
    
    // Get interpolated position
    float3 position = normalize(interpolateVec3(bc, vertices[0].pos, vertices[1].pos, vertices[2].pos));

    // Get interpolated UVs
    UVInterpolation uv = interpolateUVs(bc, vertices[0].uv, vertices[1].uv, vertices[2].uv);

    float4 color = material_data.color;
    if (material_data.textureIdx != -1) {
        color *= materialTexturesArray[material_data.textureIdx].SampleLevel(
            linearSampler, uv.interp, 0);
    }

    outputBuffer[pushConst.imageIndex][target_pixel] = float4(
        color.xyz, float(zeroDummy()));
}
