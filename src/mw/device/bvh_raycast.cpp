#define MADRONA_MWGPU_MAX_BLOCKS_PER_SM 4

#include <madrona/bvh.hpp>
#include <madrona/mesh_bvh.hpp>
#include <madrona/mw_gpu/host_print.hpp>

// #define MADRONA_PROFILE_BVH_KERNEL

using namespace madrona;

namespace sm {

// Only shared memory to be used
extern __shared__ uint8_t buffer[];

struct RaycastCounters {
    cuda::atomic<uint64_t, cuda::thread_scope_block> timingCounts;
    cuda::atomic<uint64_t, cuda::thread_scope_block> tlasTime;
    cuda::atomic<uint64_t, cuda::thread_scope_block> blasTime;
    cuda::atomic<uint64_t, cuda::thread_scope_block> numTLASTraces;
    cuda::atomic<uint64_t, cuda::thread_scope_block> numBLASTraces;
};

}

extern "C" __constant__ BVHParams bvhParams;

enum class ProfilerState {
    TLAS,
    BLAS,
    None
};

// For tracing purposes.
uint64_t globalTimer()
{
    uint64_t timestamp;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(timestamp));
    return timestamp;
}

struct Profiler {
    uint64_t timeInTLAS;
    uint64_t timeInBLAS;
    uint64_t numBLASTraces;
    uint64_t numTLASTraces;
};

#define LOG(...) mwGPU::HostPrint::log(__VA_ARGS__)

inline math::Vector3 lighting(
        math::Vector3 diffuse,
        math::Vector3 normal,
        math::Vector3 raydir,
        float roughness,
        float metalness) {

    constexpr float ambient = 0.4;

    math::Vector3 lightDir = math::Vector3{0.5,0.5,0};
    return (fminf(fmaxf(normal.dot(lightDir),0.f)+ambient, 1.0f)) * diffuse;

}

struct SphereIntersection {
    bool intersects;
    float tHitClosest;
};

SphereIntersection raySphereIntersect(const math::Vector3 &sphere_center,
                                      float sphere_radius,
                                      const math::Vector3 &ray_origin,
                                      const math::Vector3 &ray_dir)
{
    math::Vector3 oc = ray_origin - sphere_center;
    float a = dot(ray_dir, ray_dir);
    float b = 2.0 * dot(oc, ray_dir);
    float c = dot(oc,oc) - sphere_radius * sphere_radius;
    float discriminant = b*b - 4*a*c;
    if(discriminant < 0){
        return {
            .intersects = false,
            .tHitClosest = -1.f
        };
    }
    else{
        return {
            .intersects = true,
            .tHitClosest = (-b - sqrt(discriminant)) / (2.0f*a)
        };
    }
}

struct HitInfo {
    uint32_t objectID;
    uint32_t speciesID;
};

static __device__ bool traceRayTLAS(uint32_t world_idx,
                                    const math::Vector3 &ray_o,
                                    math::Vector3 ray_d,
                                    float *out_hit_t,
                                    math::Vector3 *out_color,
                                    Entity *out_entity,
                                    float t_max,
                                    HitInfo *out_hit_info)
{
    static constexpr float epsilon = 0.00001f;

    if (ray_d.x == 0.f) {
        ray_d.x += epsilon;
    }

    if (ray_d.y == 0.f) {
        ray_d.y += epsilon;
    }

    if (ray_d.z == 0.f) {
        ray_d.z += epsilon;
    }

    using namespace madrona::math;

    BVHInternalData *internal_data = bvhParams.internalData;

    float near_sphere = bvhParams.nearSphere;

    // internal_nodes_offset contains the offset to instance attached
    // data of world being operated on.
    uint32_t internal_nodes_offset = bvhParams.instanceOffsets[world_idx];
    uint32_t num_instances = bvhParams.instanceCounts[world_idx];

    QBVHNode *nodes = internal_data->traversalNodes +
                      internal_nodes_offset;

    render::InstanceData *instances = bvhParams.instances + 
                                      internal_nodes_offset;

    render::TraversalStack stack = {};

    // This starts us at root
    stack.push(1);

    bool ray_hit = false;

    float t_min = bvhParams.nearSphere;

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
                    int32_t instance_idx = (int32_t)(-node->childrenIdx[i] - 1);

                    if (instance_idx >= num_instances)
                        LOG("Got incorrect instance index: {}\n", instance_idx);

                    render::InstanceData *instance_data =
                        &instances[instance_idx];

                    if (instance_data->objectIDX == 0 || instance_data->objectIDX == 2) {
                        // Get sphere intersection with this instance
                        SphereIntersection sphere_intersect_info =
                            raySphereIntersect({instance_data->position.x, 
                                    instance_data->position.y,
                                    instance_data->zOffset},
                                    instance_data->scale.x,
                                    ray_o, ray_d);

                        if (sphere_intersect_info.intersects && 
                                sphere_intersect_info.tHitClosest > t_min) {
                            ray_hit = true;

                            t_max = sphere_intersect_info.tHitClosest;
                            *out_hit_t = t_max;
                            *out_entity = instance_data->owner;

                            out_hit_info->objectID = instance_data->objectIDX;

                            if (instance_data->objectIDX == 0) {
                                out_hit_info->speciesID = instance_data->speciesIDX;
                            }
                        }
                    }
                } else {
                    stack.push(node->childrenIdx[i]);
                }
            }
        }
    }

    return ray_hit;
}


extern "C" __global__ void bvhRaycastEntry()
{
    const uint32_t num_worlds = bvhParams.numWorlds;
    const uint32_t total_num_views = bvhParams.viewOffsets[num_worlds-1] +
                                     bvhParams.viewCounts[num_worlds-1];

    const uint32_t views_per_block = blockDim.x / bvhParams.renderOutputResolution;

    // This is the number of views currently being processed.
    const uint32_t num_resident_views = gridDim.x * views_per_block;

    // Each thread deals with one pixel of the output
    const uint32_t threads_per_view = bvhParams.renderOutputResolution;

    // This is the offset into the resident view processors that we are
    // currently in.
    const uint32_t resident_view_offset = blockIdx.x * views_per_block +
        threadIdx.x / threads_per_view;

    const uint32_t thread_offset_in_view = threadIdx.x % threads_per_view;

    uint32_t current_view_offset = resident_view_offset;

    uint32_t bytes_per_view = bvhParams.renderOutputResolution;

    while (current_view_offset < total_num_views) {
        // While we still have views to generate, trace.
        render::PerspectiveCameraData *view_data = 
            &bvhParams.views[current_view_offset];

        uint32_t world_idx = (uint32_t)view_data->worldIDX;


        // Initialize the ray and trace.
        float ray_theta = view_data->viewDirPolar;

        if (thread_offset_in_view < view_data->numForwardRays) {
            // First pixels are the forward rays
            ray_theta = (ray_theta - math::pi * 0.25f) + thread_offset_in_view *
                (math::pi * 0.5f / ((float)view_data->numForwardRays - 1.f));
        } else {
            // Last pixels are the backward rays
            ray_theta += math::pi;

            ray_theta = (ray_theta - math::pi * 0.25f) + 
                (thread_offset_in_view - view_data->numForwardRays) *
                (math::pi * 0.5f / ((float)view_data->numBackwardRays - 1.f));
        }


        math::Vector3 ray_start = { 
            view_data->position.x,
            view_data->position.y,
            view_data->zOffset 
        };

        // math::Vector3 look_at = rot.inv().rotateVec({0, 1, 0});
        math::Vector3 look_at = { cosf(ray_theta), sinf(ray_theta), 0.0f };

        // For now, just hack in a t_max of 10000.
        float t;
        math::Vector3 color;
        Entity seen_entity;
        HitInfo hit_info;
        bool hit = traceRayTLAS(
                world_idx, 
                ray_start, look_at, 
                &t, &color, &seen_entity, 10000.f,
                &hit_info);

        uint32_t linear_pixel_idx = thread_offset_in_view;

        uint32_t global_pixel_idx = view_data->rowIDX * bytes_per_view +
            linear_pixel_idx;

        uint8_t *depth_write_out = (uint8_t *)bvhParams.depthOutput + global_pixel_idx;
        int8_t *semantic_write_out = (int8_t *)bvhParams.semanticOutput + global_pixel_idx;

        if (hit) {
            if (t > 255.f) 
                depth_write_out[0] = 255;
            else
                depth_write_out[0] = (uint8_t)t;

            // We need to write the semantic information too.
            semantic_write_out[0] = (hit_info.objectIDX == 0) ?
                hit_info.speciesIDX : 0;

            // Make sure to write out the finder information
            if (thread_offset_in_view == view_data->numForwardRays / 2) {
                render::FinderOutput *entity_write_out = 
                    (render::FinderOutput *)bvhParams.finderOutput +
                        view_data->rowIDX;

                *entity_write_out = {
                    seen_entity, t
                };
            }
        } else {
            depth_write_out[0] = 255;
            semantic_write_out[0] = -1;

            if (thread_offset_in_view == view_data->numForwardRays / 2) {
                render::FinderOutput *entity_write_out = 
                    (render::FinderOutput *)bvhParams.finderOutput +
                        view_data->rowIDX;

                *entity_write_out = {
                    Entity::none(),
                    255.f
                };
            }
        }

#if 0
        // As sanity check make sure the output gets written to the gith image.
        if (linear_pixel_idx == 0) {
            write_out[0] = (uint8_t)view_data->rowIDX;
        }
#endif

        current_view_offset += num_resident_views;

        __syncwarp();
    }
}
