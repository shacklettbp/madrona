#include "shader_common.h"
#include "utils.hlsl"

struct Triangle {
    Vertex a;
    Vertex b;
    Vertex c;
};

[[vk::push_constant]]
cbuffer PushConstant {
    RTPushConstant push_const;
};

[[vk::binding(0, 0)]]
StructuredBuffer<ViewData> viewDataBuffer;

[[vk::binding(0, 1)]]
RaytracingAccelerationStructure tlas;

[[vk::binding(0, 2)]]
StructuredBuffer<ObjectData> objectDataBuffer;

[[vk::binding(0, 3)]]
RWStructuredBuffer<uint8_t> rgbOut;

[[vk::binding(0, 4)]]
RWStructuredBuffer<float> depthOut;

Camera unpackCamera(PackedCamera packed)
{
    float aspect = float(RES_X) / float(RES_Y);

    float4 rot = packed.rotation;
    float3 right = quatRotate(rot, float3(1.f, 0.f, 0.f));
    float3 view = quatRotate(rot, float3(0.f, 1.f, 0.f));
    float3 up = quatRotate(rot, float3(0.f, 0.f, 1.f)); 

    float4 pos_fov = packed.posAndTanFOV;

    float3 origin = pos_fov.xyz;

    float right_scale = aspect * pos_fov.w;
    float up_scale = pos_fov.w;

    return Camera(origin, view, up, right, right_scale, up_scale);
}

void unpackViewData(in uint32_t view_idx, out Camera cam,
                    out uint32_t world_id)
{
    ViewData view_data = viewDataBuffer[view_idx];

    cam = unpackCamera(view_data.cam);
    world_id = view_idx / MAX_VIEWS_PER_WORLD;
}

Vertex unpackVertex(uint64_t vertex_buffer, uint32_t idx)
{
    PackedVertex packed = vk::RawBufferLoad<PackedVertex>(vertex_buffer +
         sizeof(PackedVertex) * idx, 16);

    float4 a = packed.data[0];
    float4 b = packed.data[1];

    u32float3 packed_normal_tangent = uint3(
        asuint(a.w), asuint(b.x), asuint(b.y));

    float3 normal;
    float4 tangent_and_sign;
    decodeNormalTangent(packed_normal_tangent, normal, tangent_and_sign);

    Vertex vert;
    vert.position = float3(a.x, a.y, a.z);
    vert.normal = normal;
    vert.tangentAndSign = tangent_and_sign;
    vert.uv = float2(b.z, b.w);

    return vert;
}

uint3 fetchTriangleIndices(uint64_t index_buffer,
                           uint32_t index_offset)
{
    return vk::RawBufferLoad<uint3>(
        index_buffer + index_offset * sizeof(uint32_t), 4);
}

Triangle fetchTriangle(uint64_t geo_addr,
                       uint32_t mesh_offset,
                       uint32_t tri_offset)
{
    MeshData meshdata = vk::RawBufferLoad<MeshData>(
        geo_addr + mesh_offset * sizeof(MeshData), 8);
    uint32_t index_offset = meshdata.indexOffset + tri_offset * 3;
    u32float3 indices = fetchTriangleIndices(geo_addr, index_offset);

    return Triangle(
        unpackVertex(geo_addr, meshdata.vertexOffset + indices.x),
        unpackVertex(geo_addr, meshdata.vertexOffset + indices.y),
        unpackVertex(geo_addr, meshdata.vertexOffset + indices.z));
}

#define INTERPOLATE_ATTR(a, b, c, barys) \
    (a + barys.x * (b - a) + \
     barys.y * (c - a))

float3 interpolatePosition(float3 a, float3 b, float3 c, float2 barys)
{
    return INTERPOLATE_ATTR(a, b, c, barys);
}

float3 interpolateNormal(float3 a, float3 b, float3 c, float2 barys)
{
    return INTERPOLATE_ATTR(a, b, c, barys);
}

float4 interpolateCombinedTangent(float4 a, float4 b, float4 c, float2 barys)
{
    return INTERPOLATE_ATTR(a, b, c, barys);
}

float2 interpolateUV(float2 a, float2 b, float2 c, float2 barys)
{
    return INTERPOLATE_ATTR(a, b, c, barys);
}

void computeCameraRay(in Camera camera, in u32float2 idx,
                      out float3 ray_origin, out float3 ray_dir)
{
    ray_origin = camera.origin;

#ifdef PERSPECTIVE
    float2 raster = float2(idx.x, idx.y) + float2(0.5f);

    float2 screen = float2((2.f * raster.x) / RES_X - 1,
                       (2.f * raster.y) / RES_Y - 1);

    float3 right = camera.right * camera.rightScale;
    float3 up = camera.up * camera.upScale;

    ray_dir = camera.view + up * screen.y - right * screen.x ;

    ray_dir = normalize(ray_dir);
#endif

#ifdef LIDAR
    if (idx.x < 30 && idx.y == 0) {
        float theta = 2.f * M_PI * (float(idx.x) / float(30));
        float2 xy = float2(cos(theta), sin(theta));

        ray_dir = xy.x * camera.right + xy.y * camera.view;

        ray_dir = normalize(ray_dir);
    } else {
        float2 raster = float2(idx.x, idx.y) + float2(0.5f);

        float2 screen = float2((2.f * raster.x) / RES_X - 1,
                           (2.f * raster.y) / RES_Y - 1);

        float3 right = camera.right * camera.rightScale;
        float3 up = camera.up * camera.upScale;

        ray_dir = right * screen.x + up * screen.y + camera.view;

        ray_dir = normalize(ray_dir);
    }
#endif
}

bool traceShadeRay(rayQueryEXT ray_query, in uint32_t world_idx,
                   in float3 ray_origin, in float3 ray_dir,
                   uint32_t visibility_mask)
{
    rayQueryInitializeEXT(ray_query, tlas,
        gl_RayFlagsCullBackFacingTrianglesEXT,
        visibility_mask, ray_origin, 0.f, ray_dir, 220);

    while (rayQueryProceedEXT(ray_query)) {
        if (rayQueryGetIntersectionTypeEXT(ray_query, false) ==
            gl_RayQueryCandidateIntersectionTriangleEXT) {

            rayQueryConfirmIntersectionEXT(ray_query);
        }
    }

    subgroupBarrier();

    return rayQueryGetIntersectionTypeEXT(ray_query, true) !=
        gl_RayQueryCommittedIntersectionNoneEXT;
}

float toSRGB(float v)
{
    if (v <= 0.00031308f) {
        return 12.92f * v;
    } else {
        return 1.055f * pow(v, (1.f / 2.4f)) - 0.055f;
    }
}

void setOutput(uint32_t rgb_offset, uint32_t depth_offset,
               float3 rgb, float depth)
{
    rgbOut[rgb_offset] =
        uint8_t(round(255.f * toSRGB(min(rgb.x, 1.f))));
    rgbOut[rgb_offset + 1] =
        uint8_t(round(255.f * toSRGB(min(rgb.y, 1.f))));
    rgbOut[rgb_offset + 2] =
        uint8_t(round(255.f * toSRGB(min(rgb.z, 1.f))));
    rgbOut[rgb_offset + 3] = uint8_t(255);

    depthOut[depth_offset] = depth;
}

void getHitParams(in rayQueryEXT ray_query, out float2 barys,
                  out uint32_t tri_idx, out uint32_t geo_idx,
                  out uint32_t obj_idx, out uint32_t material_idx,
                  out mat4x3 o2w, out mat4x3 w2o)
{
    barys = rayQueryGetIntersectionBarycentricsEXT(ray_query, true);

    tri_idx =
        uint32_t(rayQueryGetIntersectionPrimitiveIndexEXT(ray_query, true));

    geo_idx = 
        uint32_t(rayQueryGetIntersectionGeometryIndexEXT(ray_query, true));

    obj_idx = uint32_t(
        rayQueryGetIntersectionInstanceCustomIndexEXT(ray_query, true));

    material_idx = uint32_t(
        rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT(
            ray_query, true));

    o2w = rayQueryGetIntersectionObjectToWorldEXT(ray_query, true);
    w2o = rayQueryGetIntersectionWorldToObjectEXT(ray_query, true);
}

// Entry point

layout (local_size_x = LOCAL_WORKGROUP_X,
        local_size_y = LOCAL_WORKGROUP_Y,
        local_size_z = LOCAL_WORKGROUP_Z) in;
void main()
{
    u32float3 idx = gl_GlobalInvocationID.xyz;

    bool oob = idx.x >= RES_X || idx.y >= RES_Y;
    idx.x = min(idx.x, RES_X - 1);
    idx.y = min(idx.y, RES_Y - 1);

    // Lookup our location within the launch grid
    uint32_t batch_idx = idx.z;

    uint32_t pixel_linear_idx =
        batch_idx * RES_Y * RES_X + idx.y * RES_X + idx.x;
    uint32_t rgb_out_offset = 4 * pixel_linear_idx;
    uint32_t depth_out_offset = pixel_linear_idx;

    Camera cam;
    uint32_t world_id;
    unpackViewData(batch_idx, cam, world_id);

    float3 ray_origin, ray_dir;
    computeCameraRay(cam, idx.xy, ray_origin, ray_dir);

    rayQueryEXT primary_query;
    bool primary_hit = traceShadeRay(primary_query, world_id,
                                     ray_origin, ray_dir, 1);

    if (!primary_hit) {
        setOutput(rgb_out_offset, depth_out_offset, float3(0, 0, 0), 0);
        return;
    }

    float2 hit_barys;
    uint32_t tri_idx, geo_idx, obj_idx, material_idx;
    mat4x3 o2w, w2o;
    getHitParams(primary_query, hit_barys, tri_idx,
                 geo_idx, obj_idx, material_idx, o2w, w2o);
    
    ObjectData object_data = objectDataBuffer[obj_idx];

    Triangle hit_tri = fetchTriangle(object_data.geoAddr, geo_idx, tri_idx);
    float3 world_a = transformPosition(o2w, hit_tri.a.position);
    float3 world_b = transformPosition(o2w, hit_tri.b.position);
    float3 world_c = transformPosition(o2w, hit_tri.c.position);
    float3 world_position =
        interpolatePosition(world_a, world_b, world_c, hit_barys);

    float3 hit_obj_normal = interpolateNormal(hit_tri.a.normal,
                                            hit_tri.b.normal,
                                            hit_tri.c.normal,
                                            hit_barys);

    float3 hit_world_normal = transformNormal(w2o, hit_obj_normal);

    float hit_angle = 
        max(dot(normalize(hit_world_normal), normalize(-ray_dir)), 0.f);

    float depth = distance(world_position, cam.origin);

    float3 rgb = float3(hit_angle);

    if (!oob) {
        setOutput(rgb_out_offset, depth_out_offset, rgb, depth);
    }
}
