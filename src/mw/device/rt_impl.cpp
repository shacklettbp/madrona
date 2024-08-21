#include <madrona/rt.hpp>
#include <madrona/state.hpp>

namespace madrona {
namespace mwGPU {
namespace rtentry {
template <auto trace_kn, typename ArchetypeT, typename ...ComponentT>
__global__ void traceEntry()
{
    using namespace madrona;
    using namespace madrona::render;

    uint32_t pixels_per_block = blockDim.x;

    const uint32_t total_num_views = bvhParams.internalData->numViews;

    // This is the number of views currently being processed.
    const uint32_t num_resident_views = gridDim.x;

    // This is the offset into the resident view processors that we are
    // currently in.
    const uint32_t resident_view_offset = blockIdx.x;

    uint32_t current_view_offset = resident_view_offset;

    uint32_t pixels_per_view =
        bvhParams.renderOutputResolution * 
        bvhParams.renderOutputResolution;

    uint32_t pixel_x = blockIdx.y * pixels_per_block + threadIdx.x;
    uint32_t pixel_y = blockIdx.z * pixels_per_block + threadIdx.y;

    StateManager *state_mgr = mwGPU::getStateManager();

    // These hold the perspective data pointers
    PerspectiveCameraData *perspective_datas =
        (PerspectiveCameraData *)state_mgr->getArchetypeComponent<
            RenderCameraArchetype<Archetype>, PerspectiveCameraData>();
    
    cuda::std::tuple raw_ptrs = {
        (ComponentT::PixelT *)state_mgr->getArchetypeComponent<
            ArchetypeT,
            ComponentT>()
        ...
    };

    std::apply([&](auto ...ptrs) {
        while (current_view_offset < total_num_views) {
            // While we still have views to generate, trace.
            PerspectiveCameraData *view_data = 
                &perspective_datas[current_view_offset];

            // Local to the image
            uint32_t linear_pixel_idx =
                (pixel_x + pixel_y * bvhParams.renderOutputResolution);

            // Global to the entire image buffer
            uint32_t global_pixel_off = current_view_offset * pixels_per_view +
                linear_pixel_idx;

            TraceKernelEntry info = {
                .camData = view_data,
                .pixelX = pixel_x,
                .pixelY = pixel_y,
            };

            trace_kn(info, (ptrs + global_pixel_off)...);

            current_view_offset += num_resident_views;

            __syncwarp();
        }

        __syncthreads();
    }, raw_ptrs);
}
}

template <auto full_trace_kn, typename ArchetypeT, typename ...ComponentT>
struct RTEntryInstantiate {};

template <auto trace_kn, typename ArchetypeT, typename ...ComponentT>
struct alignas(16) RTEntry : RTEntryInstantiate<
    rtentry::traceEntry<trace_kn, ArchetypeT, ComponentT...>, 
    ArchetypeT, ComponentT...> {};
}

namespace rt {
TraceResult traceRay(const TraceInfo &trace_info)
{
    static constexpr float epsilon = 0.00001f;

    if (ray_d.x == 0.f) ray_d.x += epsilon;
    if (ray_d.y == 0.f) ray_d.y += epsilon;
    if (ray_d.z == 0.f) ray_d.z += epsilon;

    using namespace madrona::math;

    BVHInternalData *internal_data = bvhParams.internalData;

    float near_sphere = trace_info.minDepth;

    // internal_nodes_offset contains the offset to instance attached
    // data of world being operated on.
    uint32_t internal_nodes_offset = bvhParams.instanceOffsets[world_idx];
    uint32_t num_instances = bvhParams.instanceCounts[world_idx];

    QBVHNode *nodes = internal_data->traversalNodes +
                      internal_nodes_offset;

    render::InstanceData *instances = bvhParams.instances + 
                                      internal_nodes_offset;

    TraversalStack stack = {};

    // This starts us at root
    stack.push(1);

    bool ray_hit = false;

    MeshBVH::HitInfo closest_hit_info = {};

    render::InstanceData *current_instance = nullptr;

    while (stack.size > 0) {
        int32_t node_idx = stack.pop() - 1;
        QBVHNode *node = &nodes[node_idx];

        for (int i = 0; i < node->numChildren; ++i) {
            math::AABB child_aabb = node->convertToAABB(i);

            float aabb_hit_t, aabb_far_t;
            Diag3x3 inv_ray_d = { 1.f/ray_d.x, 1.f/ray_d.y, 1.f/ray_d.z };
            bool intersect_aabb = child_aabb.rayIntersects(ray_o, inv_ray_d,
                    near_sphere, t_max, aabb_hit_t, aabb_far_t);

            if (aabb_hit_t <= t_max) {
                if (node->childrenIdx[i] < 0) {
                    // This child is a leaf.
                    int32_t instance_idx = (int32_t)(-node->childrenIdx[i] - 1);

                    MeshBVH *model_bvh = bvhParams.bvhs +
                        instances[instance_idx].objectID;

                    render::InstanceData *instance_data =
                        &instances[instance_idx];

                    if (instance_data->scale.d0 == 0.0f &&
                        instance_data->scale.d1 == 0.0f &&
                        instance_data->scale.d2 == 0.0f) {
                        continue;
                    }

                    // Ok we are going to just do something stupid.
                    //
                    // Also need to bound the mesh bvh trace ray by t_max.

                    Vector3 txfm_ray_o = instance_data->scale.inv() *
                        instance_data->rotation.inv().rotateVec(
                            (ray_o - instance_data->position));

                    Vector3 txfm_ray_d = instance_data->scale.inv() *
                        instance_data->rotation.inv().rotateVec(ray_d);

                    float t_scale = txfm_ray_d.length();

                    txfm_ray_d /= t_scale;

                    MeshBVH::HitInfo hit_info = {};

                    bool leaf_hit = model_bvh->traceRay(txfm_ray_o, txfm_ray_d, 
                            &hit_info, &stack, t_max * t_scale);

                    if (leaf_hit) {
                        ray_hit = true;

                        t_max = hit_info.tHit / t_scale;

                        closest_hit_info = hit_info;
                        closest_hit_info.normal = 
                            instance_data->rotation.rotateVec(
                                instance_data->scale * closest_hit_info.normal);

                        closest_hit_info.normal = 
                            closest_hit_info.normal.normalize();

                        closest_hit_info.bvh = model_bvh;

                        current_instance = instance_data;
                    }
                } else {
                    stack.push(node->childrenIdx[i]);
                }
            }
        }
    }

    if (ray_hit) {
        return TraceResult {
            .hit = true,
            .depth = t_max,
            .materialID = closest_hit_info.bvh->getMaterialIDX(closest_hit_info),
            .normal = closest_hit_info->normal,
            .uvs = closest_hit_info->uv,
            .instance = current_instance,
            .meshBVH = closest_hit_info.bvh,
        };
    } else {
        return TraceResult {
            .hit = false
        };
    }
}

math::Vector3 sampleMaterialColor(int32_t mat_idx,
                                  MeshBVH *mesh_bvh,
                                  math::Vector2 uvs)
{
    int32_t material_idx = 
        closest_hit_info.bvh->getMaterialIDX(closest_hit_info);

    Material *mat = &bvhParams.materials[material_idx];

    // *out_color = {mat->color.x, mat->color.y, mat->color.z};
    Vector3 color = {mat->color.x, mat->color.y, mat->color.z};

    if (mat->textureIdx != -1) {
        cudaTextureObject_t *tex = &bvhParams.textures[mat->textureIdx];

        float4 sampled_color = tex2D<float4>(*tex,
            closest_hit_info.uv.x, closest_hit_info.uv.y);

        math::Vector3 tex_color = {sampled_color.x,
                                   sampled_color.y,
                                   sampled_color.z};

        color.x *= tex_color.x;
        color.y *= tex_color.y;
        color.z *= tex_color.z;
    }

    *out_color = lighting(color, closest_hit_info.normal, ray_d, 1, 1);
}

}
}
