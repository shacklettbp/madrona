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

#if 0
struct Profiler {
    uint64_t mark;
    uint64_t timeInTLAS;
    uint64_t timeInBLAS;

    // This is the number of traces we performed in the BLAS (mesh bvh tests)
    uint64_t numBLASTraces;
    // This is the number of traces we performed in the TLAS (AABB tests)
    uint64_t numTLASTraces;

    ProfilerState state;

    void markState(ProfilerState profiler_state)
    {
#ifdef MADRONA_PROFILE_BVH_KERNEL
        if (profiler_state == state) {
            return;
        } else {
            if (state == ProfilerState::None) {
                mark = globalTimer();
                state = profiler_state;
            } else if (state == ProfilerState::TLAS) {
                uint64_t new_mark = globalTimer();
                timeInTLAS += (new_mark - mark);
                mark = new_mark;
                state = profiler_state;
            } else if (state == ProfilerState::BLAS) {
                uint64_t new_mark = globalTimer();
                timeInBLAS += (new_mark - mark);
                mark = new_mark;
                state = profiler_state;
            }
        }
#endif
    }
};
#endif

#define LOG(...) mwGPU::HostPrint::log(__VA_ARGS__)

inline math::Vector3 lighting(
        math::Vector3 diffuse,
        math::Vector3 normal,
        math::Vector3 raydir,
        float roughness,
        float metalness) {

    constexpr float ambient = 0.4;
    math::Vector3 lightDir = math::Vector3{0.5,0.5,0};
    //math::Vector3 reflectDir = reflect(-lightDir, normal);
    //float spec = pow(fmaxf(dot(raydir, reflectDir), 0.0f), 32);
    return (fmaxf(normal.dot(lightDir),0.f)+ambient) * diffuse;

}

static __device__ bool traceRayTLAS(uint32_t world_idx,
                                    uint32_t view_idx,
                                    uint32_t pixel_x, uint32_t pixel_y,
                                    const math::Vector3 &ray_o,
                                    math::Vector3 ray_d,
                                    float *out_hit_t,
                                    math::Vector3 *out_color,
                                    float t_max,
                                    Profiler *profiler)
{
#define INSPECT(...) if (pixel_x == 32 && pixel_y == 32) { LOG(__VA_ARGS__); }
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

    render::MeshBVH::HitInfo closest_hit_info = {};

    uint64_t total_blas_time = 0;

#if defined(MADRONA_PROFILE_BVH_KERNEL)
    uint64_t start_time = globalTimer();
#endif

    while (stack.size > 0) {
        int32_t node_idx = stack.pop() - 1;
        QBVHNode *node = &nodes[node_idx];

#if 0
        INSPECT("Going into node: {} with {} children\n",
                node_idx, (uint32_t)node->numChildren);
#endif

        // LOG("num_children={}\n", node->numChildren);

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

                    if (instance_idx >= num_instances)
                        LOG("Got incorrect instance index: {}\n", instance_idx);

                    render::MeshBVH *model_bvh = bvhParams.bvhs +
                        instances[instance_idx].objectID;

                    render::InstanceData *instance_data =
                        &instances[instance_idx];

                    // Ok we are going to just do something stupid.
                    //
                    // Also need to bound the mesh bvh trace ray by t_max.

                    render::MeshBVH::AABBTransform txfm = {
                        instance_data->position,
                        instance_data->rotation,
                        instance_data->scale
                    };

                    Vector3 txfm_ray_o = instance_data->scale.inv() *
                        instance_data->rotation.inv().rotateVec(
                            (ray_o - instance_data->position));

                    Vector3 txfm_ray_d = instance_data->scale.inv() *
                        instance_data->rotation.inv().rotateVec(ray_d);

                    float t_scale = txfm_ray_d.length();

                    txfm_ray_d /= t_scale;

                    render::MeshBVH::HitInfo hit_info = {};

#if defined(MADRONA_PROFILE_BVH_KERNEL)
                    uint64_t blas_start_time = globalTimer();
                    bool leaf_hit = model_bvh->traceRay(txfm_ray_o, txfm_ray_d, 
                            &hit_info, &stack, txfm, t_max * t_scale);
                    uint64_t blas_end_time = globalTimer();
                    total_blas_time += (blas_end_time - blas_start_time);
#else
                    bool leaf_hit = model_bvh->traceRay(txfm_ray_o, txfm_ray_d, 
                            &hit_info, &stack, txfm, t_max * t_scale);
#endif

                    if (leaf_hit) {
                        ray_hit = true;

                        float old_t_max = t_max;
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
                    stack.push(node->childrenIdx[i]);
                }
            }
        }
    }

#if defined(MADRONA_PROFILE_BVH_KERNEL)
    uint64_t end_time = globalTimer();
    uint64_t total_trace_time = end_time - start_time;

    profiler->timeInBLAS += total_blas_time;
    profiler->timeInTLAS += (total_trace_time - total_blas_time);
#endif

    if (ray_hit) {
#if defined (MADRONA_RENDER_RGB)
        int32_t material_idx = 
            closest_hit_info.bvh->getMaterialIDX(closest_hit_info);

        Material *mat = &bvhParams.materials[material_idx];

        // *out_color = {mat->color.x, mat->color.y, mat->color.z};
        Vector3 color = {mat->color.x, mat->color.y, mat->color.z};

        if (mat->textureIdx != -1) {
            cudaTextureObject_t *tex = &bvhParams.textures[mat->textureIdx];

            float4 sampled_color = tex2D<float4>(*tex,
                closest_hit_info.uv.x, 1.f - closest_hit_info.uv.y);

            math::Vector3 tex_color = {sampled_color.x,
                                       sampled_color.y,
                                       sampled_color.z};

            color.x *= tex_color.x;
            color.y *= tex_color.y;
            color.z *= tex_color.z;
        }

        *out_color = lighting(color, closest_hit_info.normal, ray_d, 1, 1);
#else
        *out_color = closest_hit_info.normal;
#endif
    }

    return ray_hit;
}


extern "C" __global__ void bvhRaycastEntry()
{
#ifdef MADRONA_PROFILE_BVH_KERNEL
    if (threadIdx.x == 0) {
        sm::RaycastCounters *counters = (sm::RaycastCounters *)sm::buffer;
        counters->timingCounts.store(0, std::memory_order_relaxed);
        counters->tlasTime.store(0, std::memory_order_relaxed);
        counters->blasTime.store(0, std::memory_order_relaxed);
        counters->numTLASTraces.store(0, std::memory_order_relaxed);
        counters->numBLASTraces.store(0, std::memory_order_relaxed);
    }

    __syncthreads();
#endif

    uint32_t pixels_per_block = blockDim.x;

    Profiler profiler = {
        .timeInTLAS = 0,
        .timeInBLAS = 0,
        .numBLASTraces = 0,
        .numTLASTraces = 0
    };

    const uint32_t num_worlds = bvhParams.numWorlds;
    const uint32_t total_num_views = bvhParams.viewOffsets[num_worlds-1] +
                                     bvhParams.viewCounts[num_worlds-1];

    // This is the number of views currently being processed.
    const uint32_t num_resident_views = gridDim.x;

    // This is the offset into the resident view processors that we are
    // currently in.
    const uint32_t resident_view_offset = blockIdx.x;

    uint32_t current_view_offset = resident_view_offset;

    uint32_t bytes_per_view =
        bvhParams.renderOutputResolution * bvhParams.renderOutputResolution * 3;

    uint32_t num_processed_pixels = 0;

    while (current_view_offset < total_num_views) {
        // While we still have views to generate, trace.
        render::PerspectiveCameraData *view_data = 
            &bvhParams.views[current_view_offset];

        uint32_t world_idx = (uint32_t)view_data->worldIDX;


        // Initialize the ray and trace.
        math::Quat rot = view_data->rotation;
        math::Vector3 ray_start = view_data->position;
        math::Vector3 look_at = rot.inv().rotateVec({0, 1, 0});

        constexpr float theta = 1.5708f;

        const float h = tanf(theta / 2);

        const auto viewport_height = 2 * h;
        const auto viewport_width = viewport_height;
        const auto forward = look_at.normalize();

        auto u = rot.inv().rotateVec({1, 0, 0});
        auto v = cross(forward, u).normalize();

        auto horizontal = u * viewport_width;
        auto vertical = v * viewport_height;

        auto lower_left_corner = ray_start - horizontal / 2 - vertical / 2 + forward;

        uint32_t pixel_x = blockIdx.y * pixels_per_block + threadIdx.x;
        uint32_t pixel_y = blockIdx.z * pixels_per_block + threadIdx.y;

        float pixel_u = ((float)pixel_x) / (float)bvhParams.renderOutputResolution;
        float pixel_v = ((float)pixel_y) / (float)bvhParams.renderOutputResolution;

        math::Vector3 ray_dir = lower_left_corner + pixel_u * horizontal + 
            pixel_v * vertical - ray_start;
        ray_dir = ray_dir.normalize();

        float t;
#if 0
        math::Vector3 normal = {pixel_u * 2 - 1, pixel_v * 2 - 1, 0};
        normal = ray_dir;
#endif

        math::Vector3 color;

        // For now, just hack in a t_max of 10000.
        bool hit = traceRayTLAS(
                world_idx, current_view_offset, 
                pixel_x, pixel_y,
                ray_start, ray_dir, 
                &t, &color, 10000.f, &profiler);

        uint32_t linear_pixel_idx = 3 * 
            (pixel_y + pixel_x * bvhParams.renderOutputResolution);
        uint32_t global_pixel_idx = current_view_offset * bytes_per_view +
            linear_pixel_idx;

        uint8_t *write_out = (uint8_t *)bvhParams.renderOutput + global_pixel_idx;

        if (hit) {
            write_out[0] = (color.x) * 255;
            write_out[1] = (color.y) * 255;
            write_out[2] = (color.z) * 255;
        } else {
            write_out[0] = 0;
            write_out[1] = 0;
            write_out[2] = 0;
        }

        current_view_offset += num_resident_views;

        num_processed_pixels++;

        __syncwarp();
    }

    __syncthreads();

#ifdef MADRONA_PROFILE_BVH_KERNEL
    // Update the counters in shared memory first
    sm::RaycastCounters *counters = (sm::RaycastCounters *)sm::buffer;
    counters->timingCounts.fetch_add(num_processed_pixels,
            std::memory_order_relaxed);
    counters->tlasTime.fetch_add(profiler.timeInTLAS,
            std::memory_order_relaxed);
    counters->blasTime.fetch_add(profiler.timeInBLAS,
            std::memory_order_relaxed);
    counters->numTLASTraces.fetch_add(profiler.numTLASTraces,
            std::memory_order_relaxed);
    counters->numBLASTraces.fetch_add(profiler.numBLASTraces,
            std::memory_order_relaxed);

    __syncthreads();

    if (threadIdx.x == 0) {
        uint64_t timing_counts = counters->timingCounts.load(std::memory_order_relaxed);
        uint64_t tlas_time = counters->tlasTime.load(std::memory_order_relaxed);
        uint64_t blas_time = counters->blasTime.load(std::memory_order_relaxed);
        uint64_t num_tlas_traces = counters->numTLASTraces.load(std::memory_order_relaxed);
        uint64_t num_blas_traces = counters->numBLASTraces.load(std::memory_order_relaxed);

        bvhParams.timingInfo->timingCounts.fetch_add_relaxed(num_processed_pixels);
        bvhParams.timingInfo->tlasTime.fetch_add_relaxed(profiler.timeInTLAS);
        bvhParams.timingInfo->blasTime.fetch_add_relaxed(profiler.timeInBLAS);
        bvhParams.timingInfo->numTLASTraces.fetch_add_relaxed(profiler.numTLASTraces);
        bvhParams.timingInfo->numBLASTraces.fetch_add_relaxed(profiler.numBLASTraces);
    }
#endif
}
