#include <metal_stdlib>
using namespace metal;

#include "shader_common.h"

struct v2f {
    float4 position [[position]];
    float3 viewDir;
    float3 normal;
    float2 uv;
    uint viewIdx [[render_target_array_index]];
};

struct alignas(16) Quat {
    float w;
    float x;
    float y;
    float z;

    friend inline Quat operator*(Quat a, Quat b)
    {
        return Quat {
            (a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z),
            (a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y),
            (a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x),
            (a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w),
        };
    }

    inline float3 rotateVec(float3 v) const
    {
        float3 pure {x, y, z};
        float scalar = w;

        float3 pure_x_v = cross(pure, v);
        float3 pure_x_pure_x_v = cross(pure, pure_x_v);
        
        return v + 2.f * ((pure_x_v * scalar) + pure_x_pure_x_v);
    }

};

void computeTransforms(float3 obj_t,
                       float4 obj_r_raw,
                       float3 obj_s,
                       PerspectiveCameraData cam,
                       thread float4x4 &o2s,
                       thread float3x3 &o2v_normal)
{
    Quat obj_r { 
        obj_r_raw.x,
        obj_r_raw.y,
        obj_r_raw.z,
        obj_r_raw.w,
    };

    Quat cam_r_inv = {
        cam.rotation.x,
        cam.rotation.y,
        cam.rotation.z,
        cam.rotation.w,
    };

    float3 t = cam_r_inv.rotateVec(obj_t - cam.position);
    Quat r = cam_r_inv * obj_r;

    float x2 = r.x * r.x;
    float y2 = r.y * r.y;
    float z2 = r.z * r.z;
    float xz = r.x * r.z;
    float xy = r.x * r.y;
    float yz = r.y * r.z;
    float wx = r.w * r.x;
    float wy = r.w * r.y;
    float wz = r.w * r.z;

    float x2_p_y2 = x2 + y2;
    float x2_p_z2 = x2 + z2;
    float y2_p_z2 = y2 + z2;
    float xy_p_wz = xy + wz;
    float xz_p_wy = xz + wy;
    float yz_p_wx = yz + wx;
    float xz_m_wy = xz - wy;
    float xy_m_wz = xy - wz;
    float yz_m_wx = yz - wx;

    float3 s = obj_s;
    float3 ds = 2.f * s;

    o2s = float4x4 {
        float4 {
            cam.xScale * (s.x - ds.x * y2_p_z2),
            cam.yScale * ds.x * xz_m_wy,
            0.f,
            ds.x * xy_p_wz,
        },
        float4 {
            cam.xScale * ds.y * xy_m_wz,
            cam.yScale * ds.y * yz_p_wx,
            0.f,
            s.y - ds.y * x2_p_z2,
        },
        float4 {
            cam.xScale * ds.z * xz_p_wy,
            cam.yScale * (s.z - ds.z * x2_p_y2),
            0.f,
            ds.z * yz_m_wx,
        },
        float4 {
            cam.xScale * t.x,
            cam.yScale * t.z,
            cam.zNear,
            t.y,
        },
    };

    float3 normal_s = 1.f / obj_s;
    float3 normal_ds = 2.f * normal_s;

    o2v_normal = float3x3 {
        float3 {
            normal_s.x - normal_ds.x * y2_p_z2,
            normal_ds.x * xy_p_wz,
            normal_ds.x * xz_m_wy,
        },
        float3 {
            normal_ds.y * xy_m_wz,
            normal_s.y - normal_ds.y * x2_p_z2,
            normal_ds.y * yz_p_wx,
        },
        float3 {
            normal_ds.z * xz_p_wy,
            normal_ds.z * yz_m_wx,
            normal_s.z - normal_ds.z * x2_p_y2,
        },
    };
}

[[max_total_threads_per_threadgroup(consts::threadsPerInstance)]]
kernel void setupMultiview(
    constant RenderArgBuffer &render_args [[buffer(0)]],
    constant AssetsArgBuffer &asset_args [[buffer(1)]], 
    threadgroup int32_t *group_draw_idx [[threadgroup(0)]],
    uint group_idx [[threadgroup_position_in_grid]],
    uint group_thread_offset [[thread_index_in_threadgroup]])
{
    uint instance_idx = group_idx;

    constant InstanceData &instance_data =
        render_args.engineInstances[instance_idx];

    uint32_t obj_idx = instance_data.objectID;
    uint32_t world_idx = instance_data.worldID;

    int32_t num_views = render_args.numViews[world_idx];

    constant ObjectData &obj_data = asset_args.objects[obj_idx];
    int32_t num_draws = num_views * obj_data.numMeshes;

    int32_t base_draw_offset;
    if (group_thread_offset == 0) {
        base_draw_offset =
            atomic_fetch_add_explicit(render_args.numDraws, num_draws,
                                      memory_order_relaxed);
        *group_draw_idx = base_draw_offset;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    base_draw_offset = *group_draw_idx;

    for (int32_t i = 0; i < num_views; i += consts::threadsPerInstance) {
        int32_t view_idx = group_thread_offset + i;
        if (view_idx >= num_views) {
            break;
        }

        int32_t view_layer_idx =
            world_idx * render_args.numMaxViewsPerWorld + view_idx;

        PerspectiveCameraData cam_data = render_args.views[view_layer_idx];

        float4x4 o2s;
        float3x3 o2v_normal;
        computeTransforms(instance_data.position,
                          instance_data.rotation,
                          instance_data.scale,
                          cam_data,
                          o2s, o2v_normal);


        int32_t draw_offset = base_draw_offset + view_idx * obj_data.numMeshes;

        // TODO: split transform calc and instance output and save transforms in shared mem

        for (int32_t mesh_idx = 0; mesh_idx < obj_data.numMeshes; mesh_idx++) {
            int32_t draw_idx = draw_offset + mesh_idx;

            render_args.drawInstances[draw_idx] = DrawInstanceData {
                o2s,
                o2v_normal,
                view_layer_idx,
                {},
            };

            render_command draw_cmd(render_args.drawICB, draw_idx);

            constant MeshData &mesh_data =
                asset_args.meshes[obj_data.meshOffset + mesh_idx];
            
            constant uint *mesh_indices =
                asset_args.indices + mesh_data.indexOffset;

            draw_cmd.draw_indexed_primitives(
                primitive_type::triangle,
                mesh_data.numIndices,
                mesh_indices,
                1,
                mesh_data.vertexOffset,
                draw_idx);
        }
    }
}

float3 octahedralVectorDecode(float2 f)
{
     f = f * 2.0 - 1.0;
     // https://twitter.com/Stubbesaurus/status/937994790553227264
     float3 n(f.x, f.y, 1.f - abs(f.x) - abs(f.y));
     float t = clamp(-n.z, 0.0, 1.0);
     n.x += n.x >= 0.0 ? -t : t;
     n.y += n.y >= 0.0 ? -t : t;
     return normalize(n);
}

void decodeNormalTangent(uint3 packed, thread float3 &normal,
                         thread float4 &tangent_and_sign)
{
    float2 ab = float2(as_type<half2>(packed.x));
    float2 cd = float2(as_type<half2>(packed.y));

    normal = float3(ab.x, ab.y, cd.x);
    float sign = cd.y;

    float2 oct_tan = unpack_snorm2x16_to_float(packed.z);
    float3 tangent = octahedralVectorDecode(oct_tan);

    tangent_and_sign = float4(tangent, sign);
}

Vertex unpackVertex(PackedVertex packed)
{
    float4 a = packed.data[0];
    float4 b = packed.data[1];

    uint3 packed_normal_tangent(
        as_type<uint>(a.w), as_type<uint>(b.x), as_type<uint>(b.y));

    Vertex vert;
    vert.position = float3(a.x, a.y, a.z);
    decodeNormalTangent(packed_normal_tangent, vert.normal,
                        vert.tangentAndSign);
    vert.uv = float2(b.z, b.w);

    return vert;
}

v2f vertex vertMain(uint vert_idx [[vertex_id]],
                    uint instance_idx [[instance_id]],
                    constant RenderArgBuffer &render_args [[buffer(0)]],
                    constant AssetsArgBuffer &asset_args [[buffer(1)]])
{
    PackedVertex packed_vertex = asset_args.vertices[vert_idx];
    Vertex vert = unpackVertex(packed_vertex);

    DrawInstanceData draw_instance = render_args.drawInstances[instance_idx];
    uint view_idx = draw_instance.viewIdx;

    Quat view_rot {
        render_args.views[view_idx].rotation.x,
        render_args.views[view_idx].rotation.y,
        render_args.views[view_idx].rotation.z,
        render_args.views[view_idx].rotation.w,
    };

    v2f o;
    o.position = draw_instance.objectToScreen * float4(vert.position, 1.f);
    o.viewDir = normalize(view_rot.rotateVec(
        vert.position - render_args.views[view_idx].position));
    o.normal = normalize(draw_instance.objectNormalToView * vert.normal);
    o.uv = vert.uv;
    o.viewIdx = view_idx;

    return o;
}

half4 fragment fragMain(v2f in [[stage_in]])
{
    //float hit_angle = max(dot(normalize(in.normal), to_frag), 0.f);
    float hit_angle = abs(dot(normalize(in.normal), normalize(in.viewDir)));

    return half4(half3(hit_angle), 1.0);
}

#if 0
struct RasterizerPrimData {
    uint layer [[render_target_array_index]];
};

using RasterizerData = metal::mesh<v2f,
                                   RasterizerPrimData,
                                   consts::numMaxMeshletVertices,
                                   consts::numMaxMeshletTris,
                                   topology::triangle>;

[[object,
  max_total_threads_per_threadgroup(consts::threadsPerObject)]]
  max_total_threadgroups_per_mesh_grid(consts::maxMeshletGroups)]]
void objectMain(object_data ObjectToMeshPayload &payload [[payload]],
                mesh_grid_properties mesh_grid_props,
                constant WorldDataArgs &world_args [[buffer(0)]],
                constant AssetsArgBuffer &asset_args [[buffer(1)]], 
                uint2 thread_idx [[thread_position_in_grid]])
{
    uint32_t instance_idx = thread_idx.x;
    uint32_t view_idx = thread_idx.y;
    constant InstanceData &instance_data = world_args.instanceData[instance_idx];
    float4x3 o2w = instance_data.txfm;
    uint32_t obj_idx = instance_data.objectID;

    uint32_t world_idx = instance_data.worldID;
    int32_t num_views = world_args.numViews[world_idx];
    if (view_idx < num_views) {
        mesh_grid_props.set_threadgroups_per_grid(uint3(0, 0, 0));
        return;
    }

    int32_t view_layer_idx = world_idx * world_args.numMaxViewsPerWorld + view_idx;
    float4x4 view_txfm = world_args.viewTransforms[view_layer_idx];

    float4x4 o2s = view_txfm * float4x4(o2w, 0, 0, 0, 1);

    payload.o2s = o2s;
    payload.viewLayerIdx = view_layer_idx;

    const ObjectData &obj_data = world_args.objects[obj_id];

    int32_t total_num_meshlets = 0;
    for (int32_t mesh_idx = 0; mesh_idx < obj_data.numMeshes; mesh_idx++) {
        MeshData &mesh_data = asset_args.meshes[obj_data.meshOffset + i];

        int32_t num_meshlets = min
        for (int32_t i = 0; i < 
        payload.meshletData[total_num_meshes++] = 
            ObjectToMeshPayload::MeshletData {
                mesh_data.vertexOffset,
                mesh_data.numVertices,
                mesh_data.vertexOffset,
                mesh_data.numVertices,
            };
    }

    mesh_grid_props.set_threadgroups_per_grid(uint3(obj_data.numMeshes, 0, 0));
}

[[mesh,
  max_total_threads_per_threadgroup(consts::threadsPerMeshlet)]]
void meshMain(RasterizerData raster_out,
              object_data ObjectToMeshPayload &payload [[payload]],
              constant AssetsArgBuffer &asset_args [[buffer(1)]])
{
}
#endif

