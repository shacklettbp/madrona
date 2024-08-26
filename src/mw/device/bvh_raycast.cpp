#if 1
#define MADRONA_MWGPU_MAX_BLOCKS_PER_SM 4

#include <madrona/bvh.hpp>
#include <madrona/mesh_bvh.hpp>
#include <madrona/mw_gpu/host_print.hpp>

#define LOG(...) mwGPU::HostPrint::log(__VA_ARGS__)

using namespace madrona;

namespace sm {
// Only shared memory to be used
extern __shared__ uint8_t buffer[];
}

extern "C" __constant__ BVHParams bvhParams;

inline math::Vector3 lighting(
        math::Vector3 diffuse,
        math::Vector3 normal,
        math::Vector3 raydir,
        float roughness,
        float metalness)
{
    constexpr float ambient = 0.4;

    math::Vector3 lightDir = math::Vector3{0.5,0.5,0};
    return (fminf(fmaxf(normal.dot(lightDir),0.f)+ambient, 1.0f)) * diffuse;
}

inline math::Vector3 calculateOutRay(
        render::PerspectiveCameraData *view_data,
        uint32_t pixel_x, uint32_t pixel_y)
{
    math::Quat rot = view_data->rotation;
    math::Vector3 ray_start = view_data->position;
    math::Vector3 look_at = rot.inv().rotateVec({0, 1, 0});
 
    // const float h = tanf(theta / 2);
    const float h = 1.0f / (-view_data->yScale);

    const auto viewport_height = 2 * h;
    const auto viewport_width = viewport_height;
    const auto forward = look_at.normalize();

    auto u = rot.inv().rotateVec({1, 0, 0});
    auto v = cross(forward, u).normalize();

    auto horizontal = u * viewport_width;
    auto vertical = v * viewport_height;

    auto lower_left_corner = ray_start - horizontal / 2 - vertical / 2 + forward;
  
    float pixel_u = ((float)pixel_x + 0.5f) / (float)bvhParams.renderOutputResolution;
    float pixel_v = ((float)pixel_y + 0.5f) / (float)bvhParams.renderOutputResolution;

    math::Vector3 ray_dir = lower_left_corner + pixel_u * horizontal + 
        pixel_v * vertical - ray_start;
    ray_dir = ray_dir.normalize();

    return ray_dir;
}

static __device__ bool traceRayTLAS(uint32_t world_idx,
                                    uint32_t view_idx,
                                    const math::Vector3 &ray_o,
                                    math::Vector3 ray_d,
                                    float *out_hit_t,
                                    math::Vector3 *out_color,
                                    float t_max)
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

    QBVHNode *nodes = internal_data->traversalNodes +
                      internal_nodes_offset;

    render::InstanceData *instances = bvhParams.instances + 
                                      internal_nodes_offset;


    int32_t stack[32];
    int32_t stack_size = 0;
    stack[stack_size++] = 1;


    bool ray_hit = false;

    MeshBVH::HitInfo closest_hit_info = {};


    while (stack_size > 0) {
        int32_t node_idx = stack[--stack_size] - 1;
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
                            &hit_info, stack, stack_size, t_max * t_scale);

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
                    }
                } else {
                    // stack.push(node->childrenIdx[i]);
                    stack[stack_size++] = node->childrenIdx[i];
                }
            }
        }
    }

    if (ray_hit) {
        if (bvhParams.raycastRGBD) {
            int32_t material_idx = 
                closest_hit_info.bvh->getMaterialIDX(closest_hit_info);

            Material *mat = &bvhParams.materials[material_idx];

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
        
        // *out_color = closest_hit_info.normal;
        *out_hit_t = t_max;
    }

    return ray_hit;
}

static __device__ void writeRGB(uint32_t pixel_byte_offset,
                           const math::Vector3 &color)
{
    uint8_t *rgb_out = (uint8_t *)bvhParams.rgbOutput + pixel_byte_offset;

    *(rgb_out + 0) = (color.x) * 255;
    *(rgb_out + 1) = (color.y) * 255;
    *(rgb_out + 2) = (color.z) * 255;
    *(rgb_out + 3) = 255;
}

static __device__ void writeDepth(uint32_t pixel_byte_offset,
                             float depth)
{
    float *depth_out = (float *)
        ((uint8_t *)bvhParams.depthOutput + pixel_byte_offset);
    *depth_out = depth;
}

extern "C" __global__ void bvhRaycastEntry()
{
    uint32_t pixels_per_block = blockDim.x;

    const uint32_t total_num_views = bvhParams.internalData->numViews;

    // This is the number of views currently being processed.
    const uint32_t num_resident_views = gridDim.x;

    // This is the offset into the resident view processors that we are
    // currently in.
    const uint32_t resident_view_offset = blockIdx.x;

    uint32_t current_view_offset = resident_view_offset;

    uint32_t bytes_per_view =
        bvhParams.renderOutputResolution * bvhParams.renderOutputResolution * 4;

    uint32_t num_processed_pixels = 0;

    uint32_t pixel_x = blockIdx.y * pixels_per_block + threadIdx.x;
    uint32_t pixel_y = blockIdx.z * pixels_per_block + threadIdx.y;

    while (current_view_offset < total_num_views) {
        // While we still have views to generate, trace.
        render::PerspectiveCameraData *view_data = 
            &bvhParams.views[current_view_offset];

        uint32_t world_idx = (uint32_t)view_data->worldIDX;

        math::Vector3 ray_start = view_data->position;
        math::Vector3 ray_dir = calculateOutRay(view_data, pixel_x, pixel_y);

        float t;

        math::Vector3 color;

        // For now, just hack in a t_max of 10000.
        bool hit = traceRayTLAS(
                world_idx, current_view_offset, 
                ray_start, ray_dir, 
                &t, &color, 10000.f);

        uint32_t linear_pixel_idx = 4 * 
            (pixel_y + pixel_x * bvhParams.renderOutputResolution);

        uint32_t global_pixel_byte_off = current_view_offset * bytes_per_view +
            linear_pixel_idx;

        if (bvhParams.raycastRGBD) {
            // Write both depth and color information
            if (hit) {
                writeRGB(global_pixel_byte_off, color);
                writeDepth(global_pixel_byte_off, t);
            } else {
                writeRGB(global_pixel_byte_off, { 0.f, 0.f, 0.f });
                writeDepth(global_pixel_byte_off, 0.f);
            }
        } else {
            // Only write depth information
            if (hit) {
                writeDepth(global_pixel_byte_off, t);
            } else {
                writeDepth(global_pixel_byte_off, 0.f);
            }
        }

        current_view_offset += num_resident_views;

        num_processed_pixels++;

        __syncwarp();
    }
}
#else

#define MADRONA_MWGPU_MAX_BLOCKS_PER_SM 4

#include <madrona/bvh.hpp>
#include <madrona/mesh_bvh.hpp>
#include <madrona/mw_gpu/host_print.hpp>

#define LOG(...) mwGPU::HostPrint::log(__VA_ARGS__)

using namespace madrona;
using namespace madrona::math;

namespace sm {
// Only shared memory to be used
extern __shared__ uint8_t buffer[];
}

extern "C" __constant__ BVHParams bvhParams;

static __device__ math::Vector3 calculateOutRay(
        render::PerspectiveCameraData *view_data,
        uint32_t pixel_x, uint32_t pixel_y)
{
    math::Quat rot = view_data->rotation;
    math::Vector3 ray_start = view_data->position;
    math::Vector3 look_at = rot.inv().rotateVec({0, 1, 0});
 
    // const float h = tanf(theta / 2);
    const float h = 1.0f / (-view_data->yScale);

    const auto viewport_height = 2 * h;
    const auto viewport_width = viewport_height;
    const auto forward = look_at.normalize();

    auto u = rot.inv().rotateVec({1, 0, 0});
    auto v = cross(forward, u).normalize();

    auto horizontal = u * viewport_width;
    auto vertical = v * viewport_height;

    auto lower_left_corner = ray_start - horizontal / 2 - vertical / 2 + forward;
  
    float pixel_u = ((float)pixel_x + 0.5f) / (float)128;
    float pixel_v = ((float)pixel_y + 0.5f) / (float)128;

    math::Vector3 ray_dir = lower_left_corner + pixel_u * horizontal + 
        pixel_v * vertical - ray_start;
    ray_dir = ray_dir.normalize();

    return ray_dir;
}

static __device__ void writeRGB(uint32_t pixel_byte_offset,
                           const math::Vector3 &color,
                                uint32_t debug_a,
                                uint32_t debug_b)
{
    uint8_t *rgb_out = (uint8_t *)bvhParams.rgbOutput + pixel_byte_offset;

    *(rgb_out + 0) = (color.x) * 255;
    *(rgb_out + 1) = (uint8_t)debug_a;
    *(rgb_out + 2) = (uint8_t)debug_b;
    *(rgb_out + 3) = (uint8_t)pixel_byte_offset;
}

static __device__ void writeDepth(uint32_t pixel_byte_offset,
                             float depth)
{
    float *depth_out = (float *)
        ((uint8_t *)bvhParams.depthOutput + pixel_byte_offset);
    *depth_out = depth;
}

struct TraceResult {
    bool hit;
    Vector3 color;
    float depth;
};

static __device__ TraceResult traceRayTLAS(
        uint32_t world_idx,
        math::Vector3 ray_o,
        math::Vector3 ray_d,
        float t_max)
{
    { // Adjust ray direction
        static constexpr float epsilon = 0.00001f;
        if (ray_d.x == 0.f) ray_d.x += epsilon;
        if (ray_d.y == 0.f) ray_d.y += epsilon;
        if (ray_d.z == 0.f) ray_d.z += epsilon;
    }

    TraceResult result;
    result.hit = false;

#if 0
    int32_t stack[32];
    int32_t stack_size = 0;

    stack[stack_size++] = 1;
#endif

#if 1
    TraversalStack stack;
    stack.size = 0;

    { // Initialize the stack
        stack.push(1);
    }
#endif

    // Get some required pointers and info
    const BVHInternalData *internal_data = bvhParams.internalData;
    uint32_t internal_nodes_offset = bvhParams.instanceOffsets[world_idx];
    const QBVHNode *nodes = internal_data->traversalNodes +
                            internal_nodes_offset;
    const render::InstanceData *instances = bvhParams.instances + 
                                            internal_nodes_offset;
    float near_sphere = bvhParams.nearSphere;

    Diag3x3 inv_ray_d = { 1.f/ray_d.x, 1.f/ray_d.y, 1.f/ray_d.z };

    uint32_t iter = 0;
    while (stack.size > 0) {
        // int32_t node_idx = stack[--stack_size] - 1;
        int32_t node_idx = stack.pop() - 1;
        const QBVHNode *node = &nodes[node_idx];

        for (unsigned int i = 0; i < node->numChildren; ++i) {
            math::AABB child_aabb = node->convertToAABB(i);

            float aabb_hit_t = 0.f, aabb_far_t = 0.f;
            child_aabb.rayIntersects(ray_o, inv_ray_d,
                    near_sphere, t_max, aabb_hit_t, aabb_far_t);

            if (aabb_hit_t <= t_max) {
                if (node->childrenIdx[i] < 0) {
                    result.hit = true;
                    result.color = ray_d;
                    result.depth = 1.f;
                } else {
                    // stack[stack_size++] = node->childrenIdx[i];
                    stack.push(node->childrenIdx[i]);
                }
            }
        }

        ++iter;
    }

    return result;
}

extern "C" __global__ void bvhRaycastEntry()
{
    uint32_t pixels_per_block = blockDim.x;

    const uint32_t total_num_views = bvhParams.internalData->numViews;

    // This is the number of views currently being processed.
    const uint32_t num_resident_views = gridDim.x;

    // This is the offset into the resident view processors that we are
    // currently in.
    const uint32_t resident_view_offset = blockIdx.x;

    uint32_t current_view_offset = resident_view_offset;
    if (current_view_offset >= total_num_views) {
      return;
    }

    uint32_t bytes_per_view =
        128 * 128 * 4;

    uint32_t pixel_x = blockIdx.y * pixels_per_block + threadIdx.x;
    uint32_t pixel_y = blockIdx.z * pixels_per_block + threadIdx.y;

    int i = 0;
    while (current_view_offset < total_num_views) {
        // While we still have views to generate, trace.
        render::PerspectiveCameraData *view_data = 
            &bvhParams.views[current_view_offset];

        math::Vector3 ray_start = view_data->position;
        math::Vector3 ray_dir = calculateOutRay(view_data, pixel_x, pixel_y);

        TraceResult trace_res = traceRayTLAS(
                view_data->worldIDX, ray_start, ray_dir, 
                // (float)global_pixel_byte_off);
                10000.f);

        uint32_t linear_pixel_idx = 4 * 
            (pixel_x + pixel_y * 128);

        // if (current_view_offset == 0) {
            uint32_t global_pixel_byte_off = current_view_offset * bytes_per_view +
                linear_pixel_idx;

            if (bvhParams.raycastRGBD) {
                // Write both depth and color information
                if (trace_res.hit) {
                    writeRGB(global_pixel_byte_off, trace_res.color, current_view_offset, i);
                    writeDepth(global_pixel_byte_off, trace_res.depth);
                } else {
                    writeRGB(global_pixel_byte_off, { 0.f, 0.f, 0.f }, current_view_offset, i);
                    writeDepth(global_pixel_byte_off, 0.f);
                }
            } else {
                // Only write depth information
                if (trace_res.hit) {
                    writeDepth(global_pixel_byte_off, trace_res.depth);
                } else {
                    writeDepth(global_pixel_byte_off, 0.f);
                }
            }
        // }

        current_view_offset += num_resident_views;

        // __syncwarp();
        //

        i++;
    }
}
#endif
