#include <metal_stdlib>
using namespace metal;

#include "shader_common.h"

struct v2f {
    float4 position [[position]];
    float3 normal;
    float2 uv;
    uint viewIdx [[render_target_array_index]];
};

void computeTransforms(float3 t,
                       float4 r,
                       float3 s,
                       thread float4x4 &o2w_position,
                       thread float3x3 &o2w_normal)
{
    float x2 = r.x * r.x;
    float y2 = r.y * r.y;
    float z2 = r.z * r.z;
    float xz = r.x * r.z;
    float xy = r.x * r.y;
    float yz = r.y * r.z;
    float wx = r.w * r.x;
    float wy = r.w * r.y;
    float wz = r.w * r.z;

    float3 ds = 2.f * s;

    o2w_position = float4x4 {
        float4 { 
            s.x - ds.x * (y2 + z2),
            ds.x * (xy + wz),
            ds.x * (xz - wy),
            0.f,
        },
        float4 {
            ds.y * (xy - wz),
            s.y - ds.y * (x2 + z2),
            ds.y * (yz + wx),
            0.f,
        },
        float4 {
            ds.z * (xz + wy),
            ds.z * (yz - wx),
            s.z - ds.z * (x2 + y2),
            0.f,
        },
        float4(t, 1.f),
    };

    float3 s_inv = 1.f / s;
    float3 ds_inv = 2.f * s_inv;

    o2w_normal = float3x3 {
        float3 {
            s_inv.x - ds_inv.x * (y2 + z2),
            ds_inv.y * (xy + wz),
            ds_inv.z * (xz - wy),
        },
        float3 {
            ds_inv.x * (xy - wz),
            s_inv.y - ds_inv.y * (x2 + z2),
            ds_inv.z * (yz + wx),
        },
        float3 {
            ds_inv.x * (xz + wy),
            ds_inv.y * (yz - wx),
            s_inv.z - ds_inv.z * (x2 + y2),
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
    float4x4 o2w;
    float3x3 o2w_normal;
    computeTransforms(instance_data.position,
                      instance_data.rotation,
                      instance_data.scale,
                      o2w, o2w_normal);

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
        if (view_idx < num_views) {
            continue;
        }

        int32_t draw_offset = base_draw_offset + view_idx * obj_data.numMeshes;

        int32_t view_layer_idx =
            world_idx * render_args.numMaxViewsPerWorld + view_idx;
        float4x4 view_txfm = render_args.viewTransforms[view_layer_idx];
        float4x4 object_to_screen = view_txfm * o2w;

        float3x3 view_txfm_vec {
            float3(view_txfm[0]),
            float3(view_txfm[1]),
            float3(view_txfm[2]),
        };
        float3x3 object_normal_to_screen =
            view_txfm_vec * o2w_normal;

        // TODO: split transform calc and instance output and save transforms in shared mem

        for (int32_t mesh_idx = 0; mesh_idx < obj_data.numMeshes; mesh_idx++) {
            int32_t draw_idx = draw_offset + mesh_idx;

            render_args.drawInstances[draw_idx] = DrawInstanceData {
                object_to_screen,
                object_normal_to_screen,
                view_layer_idx,
            };

            render_command draw_cmd(render_args.drawICB, draw_idx);

            constant MeshData &mesh_data =
                asset_args.meshes[obj_data.meshOffset + mesh_idx];

            draw_cmd.draw_indexed_primitives(
                primitive_type::triangle,
                mesh_data.numIndices,
                asset_args.indices + mesh_data.indexOffset,
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

    v2f o;
    o.position = draw_instance.objectToScreen * float4(vert.position, 1.f);
    o.normal = normalize(draw_instance.objectNormalToScreen * vert.normal);
    o.uv = vert.uv;
    o.viewIdx = draw_instance.viewIdx;

    return o;
}

half4 fragment fragMain(v2f in [[stage_in]])
{
    float hit_angle = max(in.normal.z, 0.f);

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

