#define MADRONA_MWGPU_MAX_BLOCKS_PER_SM 4

#include <madrona/bvh.hpp>
#include <madrona/mesh_bvh.hpp>
#include <madrona/mw_gpu/host_print.hpp>

using namespace madrona;
using namespace madrona::math;

namespace sm {
// Only shared memory to be used
extern __shared__ uint8_t buffer[];
}

extern "C" __constant__ BVHParams bvhParams;

inline Vector3 lighting(
        Vector3 diffuse,
        Vector3 normal,
        Vector3 raydir,
        float roughness,
        float metalness)
{
    constexpr float ambient = 0.4;

    Vector3 lightDir = Vector3{0.5,0.5,0};
    return (fminf(fmaxf(normal.dot(lightDir),0.f)+ambient, 1.0f)) * diffuse;
}

inline Vector3 calculateOutRay(
        render::PerspectiveCameraData *view_data,
        uint32_t pixel_x, uint32_t pixel_y)
{
    Quat rot = view_data->rotation;
    Vector3 ray_start = view_data->position;
    Vector3 look_at = rot.inv().rotateVec({0, 1, 0});
 
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

    Vector3 ray_dir = lower_left_corner + pixel_u * horizontal + 
        pixel_v * vertical - ray_start;
    ray_dir = ray_dir.normalize();

    return ray_dir;
}

static __device__ Vector3 getRayInv(const Vector3 &ray_d)
{
    float inv_diveps = 1.f / 0.0000001f;

    float ray_x_inv = copysignf(ray_d.x == 0 ? 
            inv_diveps : 1 / ray_d.x, ray_d.x);
    float ray_y_inv = copysignf(ray_d.y == 0 ?
            inv_diveps : 1 / ray_d.y, ray_d.y);
    float ray_z_inv = copysignf(ray_d.z == 0 ?
            inv_diveps : 1 / ray_d.z, ray_d.z);
    return { ray_x_inv, ray_y_inv, ray_z_inv };
}

static __device__ inline Vector3 getDirQuant(QBVHNode *node,
                                             const Vector3 &ray_inv)
{
    float dir_quant_x = __uint_as_float((node->expX + 127) << 23) * 
        ray_inv.x;
    float dir_quant_y = __uint_as_float((node->expY + 127) << 23) *
        ray_inv.y;
    float dir_quant_z = __uint_as_float((node->expZ + 127) << 23) *
        ray_inv.z;
    return { dir_quant_x, dir_quant_y, dir_quant_z };
}

static __device__ inline Vector3 getOriginQuant(QBVHNode *node,
                                                const Vector3 &ray_o,
                                                const Vector3 &ray_inv)
{
    float origin_quant_x = (node->minPoint.x - ray_o.x) * 
        ray_inv.x;
    float origin_quant_y = (node->minPoint.y - ray_o.y) * 
        ray_inv.y;
    float origin_quant_z = (node->minPoint.z - ray_o.z) * 
        ray_inv.z;
    return { origin_quant_x, origin_quant_y, origin_quant_z };
}

static __device__ inline Vector3 getTNear(QBVHNode *node,
                                          const Vector3 &dir_quant,
                                          const Vector3 &origin_quant)
{
    float t_near_x = node->qMinX[i] * dir_quant.x + origin_quant.x;
    float t_near_y = node->qMinY[i] * dir_quant.y + origin_quant.y;
    float t_near_z = node->qMinZ[i] * dir_quant.z + origin_quant.z;
    return { t_near_x, t_near_y, t_near_z };
}

static __device__ inline Vector3 getTFar(QBVHNode *node,
                                         const Vector3 &dir_quant,
                                         const Vector3 &origin_quant)
{
    float t_far_x = node->qMaxX[i] * dir_quant.x + origin_quant.x;
    float t_far_y = node->qMaxY[i] * dir_quant.y + origin_quant.y;
    float t_far_z = node->qMaxZ[i] * dir_quant.z + origin_quant.z;
    return { t_far_x, t_far_y, t_far_z };
}

static __device__ inline bool instanceIsValid(render::InstanceData *inst)
{
    return !(instance_data->scale.d0 == 0.0f &&
             instance_data->scale.d1 == 0.0f &&
             instance_data->scale.d2 == 0.0f);
}

static __device__ bool traceRayTLAS(uint32_t world_idx,
                                    uint32_t view_idx,
                                    Vector3 ray_o,
                                    Vector3 ray_d,
                                    float *out_hit_t,
                                    Vector3 *out_color,
                                    float t_max)
{
    Vector3 ray_o_world = ray_o;
    Vector3 ray_d_world = ray_d;
    float t_scale = 1.f;

    static constexpr float epsilon = 0.00001f;

    if (ray_d.x == 0.f) ray_d.x += epsilon;
    if (ray_d.y == 0.f) ray_d.y += epsilon;
    if (ray_d.z == 0.f) ray_d.z += epsilon;

    using namespace madrona::math;

    BVHInternalData *internal_data = bvhParams.internalData;

    float near_sphere = bvhParams.nearSphere;

    // internal_nodes_offset contains the offset to instance attached
    // data of world being operated on.
    uint32_t internal_nodes_offset = bvhParams.instanceOffsets[world_idx];
    uint32_t num_instances = bvhParams.instanceCounts[world_idx];

    QBVHNode *tlas_nodes = internal_data->traversalNodes +
                           internal_nodes_offset;

    render::InstanceData *instances = bvhParams.instances + 
                                      internal_nodes_offset;

    TraversalStack stack = {};

    // This starts us at root
    stack.push(0, true);

    bool ray_hit = false;

    Vector3 ray_inv = getRayInv(ray_d);

    // For BLAS
    MeshBVH::RayIsectTxfm tri_isect_txfm;
    MeshBVH *blas = nullptr;

    while (stack.size > 0) {
        // This variable holds information as to whether we are in BLAS or not
        TraversalStack::Entry node_idx = stack.pop();

        QBVHNode *node = nullptr;

        // Fetch node from the right location
        if (TraversalStack::entryIsTLAS(node_idx)) {
            node = &tlas_nodes[node_idx & (~0x80000000)];
        } else {
            node = &blas->nodes[node_idx];
        }

        Vector3 dir_quant = getDirQuant(node, ray_inv);
        Vector3 origin_quant = getOriginQuant(node, ray_o, ray_inv);

        for (int child_idx = 0; child_idx < node->numChildren; ++child_idx) {
            Vector3 t_near = getTNear(node, dir_quant, origin_quant);
            Vector3 t_far = getTFar(node, dir_quant, origin_quant);

            float t_near = fmaxf(
                    fminf(t_near.x, t_far.x), 
                    fmaxf(fminf(t_near.y, t_far.y),
                        fmaxf(fminf(t_near.z, t_far.z), 0.f)));

            float t_far = fminf(
                    fmaxf(t_far.x, t_near.x), 
                    fminf(fmaxf(t_far.y, t_near.y),
                        fminf(fmaxf(t_far.z, t_near.z), t_max)));

            if (t_near <= t_far) {
                // Intersection happened
                if (node->isLeaf(child_idx)) {
                    // If this is a leaf, we're going to have to switch from TLAS
                    // to BLAS, or from BLAS to triangle intersection test.
                    if (TraversalStack::entryIsTLAS(node_idx)) {
                        // We hit a leaf in TLAS mode. Need to transform ray
                        // direction and origin.
                        int32_t instance_idx = node->getLeafIndex(child_idx);

                        MeshBVH *blas = bvhParams.bvhs +
                                        instances[instance_idx].objectID;

                        render::InstanceData *instance_data =
                            &instances[instance_idx];

                        if (!instanceIsValid(instance_data)) {
                            continue;
                        }

                        // Transform the ray_o and ray_d for the BLAS tracing
                        ray_o = instance_data->scale.inv() *
                            instance_data->rotation.inv().rotateVec(
                                (ray_o - instance_data->position));
                        ray_d = instance_data->scale.inv() *
                            instance_data->rotation.inv().rotateVec(ray_d);
                        t_scale = ray_d.length();
                        ray_d /= t_scale;
                        t_max *= t_scale;

                        // Push the root node of the BLAS
                        stack.push(0, false /* in BLAS mode now */);
                    } else {

                    }
                } else {
                    // Just push the child if this isn't a leaf
                    stack.push(node->childrenIdx[child_idx],
                            TraversalStack::entryIsTLAS(node_idx));
                }
            }

#if 0
            if (t_near <= t_far) {
                if (node->isLeaf(i)) {
                    // This child is a leaf in the TLAS
                    int32_t instance_idx = node->getLeafIndex(i);

                    MeshBVH *model_bvh = bvhParams.bvhs +
                        instances[instance_idx].objectID;

                    render::InstanceData *instance_data =
                        &instances[instance_idx];

                    if (instance_data->scale.d0 == 0.0f &&
                        instance_data->scale.d1 == 0.0f &&
                        instance_data->scale.d2 == 0.0f) {
                        continue;
                    }

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
                    }
                } else {
                    stack.push(node->childrenIdx[i]);
                }
            }
#endif
        }
    }

    if (ray_hit) {
        if (bvhParams.raycastRGBD) {
            int32_t material_idx = 
                closest_hit_info.bvh->getMaterialIDX(closest_hit_info);

            Material *mat = &bvhParams.materials[material_idx];

            // *out_color = {mat->color.x, mat->color.y, mat->color.z};
            Vector3 color = {mat->color.x, mat->color.y, mat->color.z};

            if (mat->textureIdx != -1) {
                cudaTextureObject_t *tex = &bvhParams.textures[mat->textureIdx];

                float4 sampled_color = tex2D<float4>(*tex,
                    closest_hit_info.uv.x, closest_hit_info.uv.y);

                Vector3 tex_color = {sampled_color.x,
                                           sampled_color.y,
                                           sampled_color.z};

                color.x *= tex_color.x;
                color.y *= tex_color.y;
                color.z *= tex_color.z;
            }

            *out_color = lighting(color, closest_hit_info.normal, ray_d, 1, 1);
        }

        *out_color = closest_hit_info.normal;
        *out_hit_t = t_max;
    }

    return ray_hit;
}

static __device__ void writeRGB(uint32_t pixel_byte_offset,
                           const Vector3 &color)
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

        Vector3 ray_start = view_data->position;
        Vector3 ray_dir = calculateOutRay(view_data, pixel_x, pixel_y);

        float t;

        Vector3 color;

        // For now, just hack in a t_max of 10000.
        bool hit = traceRayTLAS(
                world_idx, current_view_offset, 
                ray_start, ray_dir, 
                &t, &color, 10000.f);

        uint32_t linear_pixel_idx = 4 * 
            (pixel_x + pixel_y * bvhParams.renderOutputResolution);
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

    __syncthreads();
}
