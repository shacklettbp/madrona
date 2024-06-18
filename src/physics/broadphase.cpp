#include <madrona/context.hpp>
#include <madrona/physics.hpp>

#include <algorithm>

#include "physics_impl.hpp"

namespace madrona::phys::broadphase {

using namespace base;
using namespace math;
using namespace geo;

BVH::BVH(const ObjectManager *obj_mgr,
         CountT max_leaves,
         float leaf_velocity_expansion,
         float leaf_accel_expansion)
    : nodes_((Node *)rawAlloc(sizeof(Node) *
                            numInternalNodes(max_leaves))),
      num_nodes_(0),
      num_allocated_nodes_(numInternalNodes(max_leaves)),
      leaf_entities_((Entity *)rawAlloc(sizeof(Entity) * max_leaves)),
      obj_mgr_(obj_mgr), // FIXME, get rid of this
      leaf_obj_ids_((ObjectID *)
                       rawAlloc(sizeof(ObjectID) * max_leaves)),
      leaf_aabbs_((AABB *)rawAlloc(sizeof(AABB) * max_leaves)),
      leaf_transforms_(
          (LeafTransform *)rawAlloc(sizeof(LeafTransform) * max_leaves)),
      leaf_parents_((uint32_t *)rawAlloc(sizeof(uint32_t) * max_leaves)),
      sorted_leaves_((int32_t *)rawAlloc(sizeof(int32_t) * max_leaves)),
      num_leaves_(0),
      num_allocated_leaves_(max_leaves),
      leaf_velocity_expansion_(leaf_velocity_expansion),
      leaf_accel_expansion_(leaf_accel_expansion),
      force_rebuild_(true)
{}

CountT BVH::numInternalNodes(CountT num_leaves) const
{
    return std::max(utils::divideRoundUp(num_leaves - 1, CountT(3)), CountT(1)) +
        num_leaves; // + num_leaves should not be necessary but the current
                    // top down build has an issue where leaves get
                    // unnecessarily split amongst internal nodes with only
                    // 1 or 2 children
}

void BVH::rebuild()
{
    int32_t num_internal_nodes = numInternalNodes(num_leaves_.load_relaxed());
    num_nodes_ = num_internal_nodes;
    assert(num_nodes_ <= num_allocated_nodes_);

    struct StackEntry {
        int32_t nodeID;
        int32_t parentID;
        int32_t offset;
        int32_t numObjs;
    };

    StackEntry stack[64];
    stack[0] = StackEntry {
        sentinel_,
        sentinel_,
        0,
        int32_t(num_leaves_.load_relaxed()),
    };

    int32_t cur_node_offset = 0;
    CountT stack_size = 1;

    while (stack_size > 0) {
        StackEntry &entry = stack[stack_size - 1];
        int32_t node_id;
        if (entry.numObjs <= 4) {
            node_id = cur_node_offset++;

            Node &node = nodes_[node_id];
            node.parentID = entry.parentID;

            for (int i = 0; i < 4; i++) {
                if (i < entry.numObjs) {
                    int32_t leaf_id = sorted_leaves_[entry.offset + i];

                    const auto &aabb = leaf_aabbs_[leaf_id];
                    leaf_parents_[leaf_id] =
                        ((uint32_t)node_id << 2) | (uint32_t)i;

                    node.setLeaf(i, leaf_id);
                    node.minX[i] = aabb.pMin.x;
                    node.minY[i] = aabb.pMin.y;
                    node.minZ[i] = aabb.pMin.z;
                    node.maxX[i] = aabb.pMax.x;
                    node.maxY[i] = aabb.pMax.y;
                    node.maxZ[i] = aabb.pMax.z;
                } else {
                    node.clearChild(i);
                    node.minX[i] = FLT_MAX;
                    node.minY[i] = FLT_MAX;
                    node.minZ[i] = FLT_MAX;
                    node.maxX[i] = -FLT_MAX;
                    node.maxY[i] = -FLT_MAX;
                    node.maxZ[i] = -FLT_MAX;
                }
            }
        } else if (entry.nodeID == sentinel_) {
            node_id = cur_node_offset++;
            // Record the node id in the stack entry for when this entry
            // is reprocessed
            entry.nodeID = node_id;

            Node &node = nodes_[node_id];
            for (CountT i = 0; i < 4; i++) {
                node.clearChild(i);
            }
            node.parentID = entry.parentID;

            // midpoint sort items
            auto midpoint_split = [this](
                    int32_t base, int32_t num_elems) {

                auto get_center = [this, base](int32_t offset) {
                    AABB aabb = leaf_aabbs_[sorted_leaves_[base + offset]];

                    return (aabb.pMin + aabb.pMax) / 2.f;
                };

                Vector3 center_min {
                    FLT_MAX,
                    FLT_MAX,
                    FLT_MAX,
                };

                Vector3 center_max {
                    -FLT_MAX,
                    -FLT_MAX,
                    -FLT_MAX,
                };

                for (int i = 0; i < num_elems; i++) {
                    const Vector3 &center = get_center(i);
                    center_min = Vector3::min(center_min, center);
                    center_max = Vector3::max(center_max, center);
                }

                auto split = [&](auto get_component) {
                    float split_val = 0.5f * (get_component(center_min) +
                                              get_component(center_max));

                    int start = 0;
                    int end = num_elems;

                    while (start < end) {
                        while (start < end &&
                               get_component(get_center(start)) < split_val) {
                            ++start;
                        }

                        while (start < end && get_component(
                                get_center(end - 1)) >= split_val) {
                            --end;
                        }

                        if (start < end) {
                            std::swap(sorted_leaves_[base + start],
                                      sorted_leaves_[base + end - 1]);
                            ++start;
                            --end;
                        }
                    }

                    if (start > 0 && start < num_elems) {
                        return start;
                    } else {
                        return num_elems / 2;
                    }
                };

                Vector3 center_diff = center_max - center_min;
                if (center_diff.x > center_diff.y &&
                    center_diff.x > center_diff.z) {
                    return split([](Vector3 v) {
                        return v.x;
                    });
                } else if (center_diff.y > center_diff.x &&
                           center_diff.y > center_diff.z) {
                    return split([](Vector3 v) {
                        return v.y;
                    });
                } else {
                    return split([](Vector3 v) {
                        return v.z;
                    });
                }
            };

            int32_t second_split = midpoint_split(entry.offset, entry.numObjs);
            int32_t num_h1 = second_split;
            int32_t num_h2 = entry.numObjs - second_split;

            int32_t first_split = midpoint_split(entry.offset, num_h1);
            int32_t third_split =
                midpoint_split(entry.offset + second_split, num_h2);

            // Setup stack to recurse into fourths. Put fourths on stack in
            // reverse order to preserve left-right depth first ordering

            stack[stack_size++] = {
                -1,
                entry.nodeID,
                entry.offset + num_h1 + third_split,
                num_h2 - third_split,
            };

            stack[stack_size++] = {
                -1,
                entry.nodeID,
                entry.offset + num_h1,
                third_split,
            };

            stack[stack_size++] = {
                -1,
                entry.nodeID,
                entry.offset + first_split,
                num_h1 - first_split,
            };

            stack[stack_size++] = {
                -1,
                entry.nodeID,
                entry.offset,
                first_split,
            };

            // Don't finish processing this node until children are processed
            continue;
        } else {
            // Revisiting this node after having processed children
            node_id = entry.nodeID;
        }

        // At this point, remove the current entry from the stack
        stack_size -= 1;

        Node &node = nodes_[node_id];
        if (node.parentID == -1) {
            continue;
        }

        AABB combined_aabb = AABB::invalid();
        for (CountT i = 0; i < 4; i++) {
            if (!node.hasChild(i)) {
                break;
            }

            combined_aabb = AABB::merge(combined_aabb, AABB {
                /* .pMin = */ {
                    node.minX[i],
                    node.minY[i],
                    node.minZ[i],
                },
                /* .pMax = */ {
                    node.maxX[i],
                    node.maxY[i],
                    node.maxZ[i],
                },
            });
        }

        Node &parent = nodes_[node.parentID];
        CountT child_offset;
        for (child_offset = 0; ; child_offset++) {
            if (parent.children[child_offset] == sentinel_) {
                break;
            }
        }

        parent.setInternal(child_offset, node_id);
        parent.minX[child_offset] = combined_aabb.pMin.x;
        parent.minY[child_offset] = combined_aabb.pMin.y;
        parent.minZ[child_offset] = combined_aabb.pMin.z;
        parent.maxX[child_offset] = combined_aabb.pMax.x;
        parent.maxY[child_offset] = combined_aabb.pMax.y;
        parent.maxZ[child_offset] = combined_aabb.pMax.z;
    }

#if 0
    {
        // validate tree bottom up
        int32_t num_leaves = num_leaves_.load_relaxed();
        for (int32_t i = 0; i < num_leaves; i++) {
            const AABB &leaf_aabb = leaf_aabbs_[i];
            uint32_t leaf_parent = leaf_parents_[i];

            int32_t node_idx = int32_t(leaf_parent >> 2_u32);
            int32_t sub_idx = int32_t(leaf_parent & 3);

            Node *node = &nodes_[node_idx];
            while (true) {
                auto invalid = [&]() {
                    printf("%d %d %d\n\t(%f %f %f) (%f %f %f)\n\t(%f %f %f) (%f %f %f)\n",
                           i, node_idx, sub_idx, leaf_aabb.pMin.x, leaf_aabb.pMin.y, leaf_aabb.pMin.z,
                           leaf_aabb.pMax.x, leaf_aabb.pMax.y, leaf_aabb.pMax.z,
                           node->minX[sub_idx], node->minY[sub_idx], node->minZ[sub_idx],
                           node->maxX[sub_idx], node->maxY[sub_idx], node->maxZ[sub_idx]);
                    assert(false);
                };

                if (leaf_aabb.pMin.x < node->minX[sub_idx]) {
                    invalid();
                }
                if (leaf_aabb.pMin.y < node->minY[sub_idx]) {
                    invalid();
                }
                if (leaf_aabb.pMin.z < node->minZ[sub_idx]) {
                    invalid();
                }

                if (leaf_aabb.pMax.x > node->maxX[sub_idx]) {
                    invalid();
                }
                if (leaf_aabb.pMax.y > node->maxY[sub_idx]) {
                    invalid();
                }
                if (leaf_aabb.pMax.z > node->maxZ[sub_idx]) {
                    invalid();
                }

                int child_idx = node_idx;
                node_idx = node->parentID;
                if (node_idx == sentinel_) {
                    break;
                }

                node = &nodes_[node_idx];

                int child_offset = -1;
                for (int j = 0; j < 4; j++) {
                    if (node->children[j] == child_idx) {
                        child_offset = j;
                        break;
                    }
                }
                sub_idx = child_offset;
            };
        }
    }

    // Validate top down
    {
        int32_t stack[128];
        AABB aabb_stack[128];
        stack[0] = 0;
        aabb_stack[0] = {
            -FLT_MAX,
            -FLT_MAX,
            -FLT_MAX,
            FLT_MAX,
            FLT_MAX,
            FLT_MAX,
        };
        CountT stack_size = 1;

        while (stack_size > 0) { 
            int32_t stack_idx = --stack_size;
            int32_t node_idx = stack[stack_idx];
            madrona::math::AABB parent_aabb = aabb_stack[stack_idx];

            const Node &node = nodes_[node_idx];
            for (int i = 0; i < 4; i++) {
                if (!node.hasChild(i)) {
                    continue; // Technically this could be break?
                }

                AABB child_aabb {
                    /* .pMin = */ {
                        node.minX[i],
                        node.minY[i],
                        node.minZ[i],
                    },
                    /* .pMax = */ {
                        node.maxX[i],
                        node.maxY[i],
                        node.maxZ[i],
                    },
                };

                auto invalid = [&]() {
                    printf("Invalid top down %d %d (%f %f %f) (%f %f %f) (%f %f %f) (%f %f %f)\n",
                           node_idx, i,
                           parent_aabb.pMin.x, parent_aabb.pMin.y, parent_aabb.pMin.z,
                           parent_aabb.pMax.x, parent_aabb.pMax.y, parent_aabb.pMax.z,
                           child_aabb.pMin.x, child_aabb.pMin.y, child_aabb.pMin.z,
                           child_aabb.pMax.x, child_aabb.pMax.y, child_aabb.pMax.z);
                    assert(false);
                };

                if (child_aabb.pMin.x < parent_aabb.pMin.x) {
                    invalid();
                }

                if (child_aabb.pMin.y < parent_aabb.pMin.y) {
                    invalid();
                }

                if (child_aabb.pMin.z < parent_aabb.pMin.z) {
                    invalid();
                }

                if (child_aabb.pMax.x > parent_aabb.pMax.x) {
                    invalid();
                }

                if (child_aabb.pMax.y > parent_aabb.pMax.y) {
                    invalid();
                }

                if (child_aabb.pMax.z > parent_aabb.pMax.z) {
                    invalid();
                }

                if (node.isLeaf(i)) {
                    int32_t leaf_idx = node.leafIDX(i);
                    if (leaf_idx >= num_leaves_.load_relaxed()) {
                        printf("Out of bounds leaf %u %u %u\n",
                               i, leaf_idx, num_leaves_.load_relaxed());
                        assert(false);
                    }
                } else {
                    stack[stack_size] = node.children[i];
                    aabb_stack[stack_size] = child_aabb;
                    stack_size += 1;
                }
            }
        }
    }
#endif
}

static inline AABB expandAABBWithMotion(
    AABB aabb,
    const Vector3 &linear_velocity,
    float velocity_expansion_factor,
    float accel_expansion_factor)
{
    // FIXME include external velocity
#pragma unroll
    for (int32_t i = 0; i < 3; i++) {
        float pos_delta =
            velocity_expansion_factor * linear_velocity[i];

        float min_delta = pos_delta - accel_expansion_factor;
        float max_delta = pos_delta + accel_expansion_factor;

        if (min_delta < 0.f) {
            aabb.pMin[i] += min_delta;
        }
        if (max_delta > 0.f) {
            aabb.pMax[i] += max_delta;
        }
    }

    return aabb;
}

void BVH::updateLeafPosition(LeafID leaf_id,
                             const Vector3 &pos,
                             const Quat &rot,
                             const Diag3x3 &scale,
                             const Vector3 &linear_vel,
                             const AABB &obj_aabb)
{
    AABB world_aabb = obj_aabb.applyTRS(pos, rot, scale);
    AABB expanded_aabb = expandAABBWithMotion(world_aabb, linear_vel,
                                              leaf_velocity_expansion_,
                                              leaf_accel_expansion_);

    leaf_aabbs_[leaf_id.id] = expanded_aabb;
    leaf_transforms_[leaf_id.id] = {
        pos,
        rot,
        scale,
    };
    sorted_leaves_[leaf_id.id] = leaf_id.id;
}

AABB BVH::expandLeaf(LeafID leaf_id,
                     const math::Vector3 &linear_vel)
{
    AABB aabb = leaf_aabbs_[leaf_id.id];
    AABB expanded_aabb = expandAABBWithMotion(aabb, linear_vel,
        leaf_velocity_expansion_, leaf_accel_expansion_);

    leaf_aabbs_[leaf_id.id] = expanded_aabb;

    return expanded_aabb;
}

MADRONA_ALWAYS_INLINE static inline float atomicMinF(float *addr, float value)
{
#ifdef MADRONA_GPU_MODE
    float old;
    if (!signbit(value)) {
        old = __int_as_float(atomicMin((int *)addr, __float_as_int(value)));
    } else {
        old = __uint_as_float(
            atomicMax((unsigned int *)addr, __float_as_uint(value)));
    }

    return old;
#else
    AtomicFloatRef a(*addr);
    float old = a.load<sync::relaxed>();

    while (old > value &&
           !a.compare_exchange_weak<sync::relaxed, sync::relaxed>(old, value))
    {}
           
    return old;
#endif
}

MADRONA_ALWAYS_INLINE static inline float atomicMaxF(float *addr, float value)
{
#ifdef MADRONA_GPU_MODE
    float old;

    // cuda::atomic::fetch_max does not seem to work properly (cuda 11.8)
    if (!signbit(value)) {
        old = __int_as_float(
            atomicMax((int *)addr, __float_as_int(value)));
    } else {
        old = __uint_as_float(
            atomicMin((unsigned int *)addr, __float_as_uint(value)));
    }

    return old;
#else
    AtomicFloatRef a(*addr);
    float old = a.load<sync::relaxed>();

    while (old < value &&
           !a.compare_exchange_weak<sync::relaxed, sync::relaxed>(old, value))
    {}
           
    return old;
#endif
}

void BVH::refitLeaf(LeafID leaf_id, const AABB &leaf_aabb)
{
    uint32_t leaf_parent = leaf_parents_[leaf_id.id];

    int32_t node_idx = int32_t(leaf_parent >> 2_u32);
    int32_t sub_idx = int32_t(leaf_parent & 3);

    Node &leaf_node = nodes_[node_idx];

    {
        auto nonAtomicMinF = [](float *ptr, float v) {
            AtomicFloatRef a(*ptr);
            float old = a.load<sync::relaxed>();
            if (v < old) {
                a.store<sync::relaxed>(v);
            }
            return old;
        };

        auto nonAtomicMaxF = [](float *ptr, float v) {
            AtomicFloatRef a(*ptr);
            float old = a.load<sync::relaxed>();
            if (v > old) {
                a.store<sync::relaxed>(v);
            }
            return old;
        };

        float x_min_prev = 
            nonAtomicMinF(&leaf_node.minX[sub_idx], leaf_aabb.pMin.x);
        float y_min_prev =
            nonAtomicMinF(&leaf_node.minY[sub_idx], leaf_aabb.pMin.y);
        float z_min_prev =
            nonAtomicMinF(&leaf_node.minZ[sub_idx], leaf_aabb.pMin.z);
        float x_max_prev =
            nonAtomicMaxF(&leaf_node.maxX[sub_idx], leaf_aabb.pMax.x);
        float y_max_prev =
            nonAtomicMaxF(&leaf_node.maxY[sub_idx], leaf_aabb.pMax.y);
        float z_max_prev =
            nonAtomicMaxF(&leaf_node.maxZ[sub_idx], leaf_aabb.pMax.z);

        bool expanded = leaf_aabb.pMin.x < x_min_prev ||
                        leaf_aabb.pMin.y < y_min_prev ||
                        leaf_aabb.pMin.z < z_min_prev ||
                        leaf_aabb.pMax.x > x_max_prev ||
                        leaf_aabb.pMax.y > y_max_prev ||
                        leaf_aabb.pMax.z > z_max_prev;

        if (!expanded) return;
    }

    int32_t child_idx = node_idx;
    node_idx = leaf_node.parentID;

    while (node_idx != sentinel_) {
        Node &node = nodes_[node_idx];
        int child_offset = -1;
        for (int j = 0; j < 4; j++) {
            if (node.children[j] == child_idx) {
                child_offset = j;
                break;
            }
        }
        assert(child_offset != -1);

        float x_min_prev =
            atomicMinF(&node.minX[child_offset], leaf_aabb.pMin.x);

        float y_min_prev =
            atomicMinF(&node.minY[child_offset], leaf_aabb.pMin.y);

        float z_min_prev =
            atomicMinF(&node.minZ[child_offset], leaf_aabb.pMin.z);

        float x_max_prev =
            atomicMaxF(&node.maxX[child_offset], leaf_aabb.pMax.x);

        float y_max_prev =
            atomicMaxF(&node.maxY[child_offset], leaf_aabb.pMax.y);

        float z_max_prev =
            atomicMaxF(&node.maxZ[child_offset], leaf_aabb.pMax.z);

        bool expanded = leaf_aabb.pMin.x < x_min_prev ||
                        leaf_aabb.pMin.y < y_min_prev ||
                        leaf_aabb.pMin.z < z_min_prev ||
                        leaf_aabb.pMax.x > x_max_prev ||
                        leaf_aabb.pMax.y > y_max_prev ||
                        leaf_aabb.pMax.z > z_max_prev;

        if (!expanded) {
            break;
        }

        child_idx = node_idx;
        node_idx = node.parentID;
    }
}

void BVH::updateTree()
{
    if (force_rebuild_) {
        force_rebuild_ = false;
        rebuild();
    }
    //rebuild();
}

Entity BVH::traceRay(Vector3 o,
                     Vector3 d,
                     float *out_hit_t,
                     Vector3 *out_hit_normal,
                     float t_max)
{
    using namespace math;

    Diag3x3 inv_d = Diag3x3::fromVec(d).inv();

    int32_t stack[32];
    stack[0] = 0;
    CountT stack_size = 1;

    Entity closest_hit_entity = Entity::none();
    Vector3 closest_hit_normal;

    while (stack_size > 0) { 
        int32_t node_idx = stack[--stack_size];
        const Node &node = nodes_[node_idx];
        for (int i = 0; i < 4; i++) {
            if (!node.hasChild(i)) {
                continue; // Technically this could be break?
            };

            madrona::math::AABB child_aabb {
                /* .pMin = */ {
                    node.minX[i],
                    node.minY[i],
                    node.minZ[i],
                },
                /* .pMax = */ {
                    node.maxX[i],
                    node.maxY[i],
                    node.maxZ[i],
                },
            };

            if (child_aabb.rayIntersects(o, inv_d, 0.f, t_max)) {
                if (node.isLeaf(i)) {
                    int32_t leaf_idx = node.leafIDX(i);
                    
                    float hit_t;
                    Vector3 leaf_hit_normal;
                    bool leaf_hit = traceRayIntoLeaf(
                        leaf_idx, o, d, 0.f, t_max, &hit_t, &leaf_hit_normal);

                    if (leaf_hit) {
                        t_max = hit_t;
                        closest_hit_entity = leaf_entities_[leaf_idx];
                        closest_hit_normal = leaf_hit_normal;
                    }
                } else {
                    stack[stack_size++] = node.children[i];
                }
            }
        }
    }
    
    if (closest_hit_entity == Entity::none()) {
        return Entity::none();
    }

    *out_hit_t = t_max;
    *out_hit_normal = closest_hit_normal;
    return closest_hit_entity;
}

static inline bool traceRayIntoPlane(
    Vector3 ray_o, Vector3 ray_d,
    float t_min, float t_max,
    float *hit_t,
    Vector3 *hit_normal)
{
    // ray_o and ray_d have already been transformed into the space of the
    // plane. normal is (0, 0, 1), d is 0

    float denom = ray_d.z;

    if (denom == 0) {
        return false;
    }

    float t = -ray_o.z / denom;

    if (t < t_min || t > t_max) {
        return false;
    }

    *hit_t = t;
    *hit_normal = Vector3 { 0, 0, 1 };
    return true;
}

// RTCD 5.3.8 (modified from segment to ray). Algorithm also in GPU Gems 2.
// Intersect ray r(t)=ray_o + , t_min <= t <=t_max against convex polyhedron
// specified by the n halfspaces defined by the planes p[]. On exit tfirst
// and tlast define the intersection, if any
static inline bool traceRayIntoConvexPolyhedron(
    const HalfEdgeMesh &convex_mesh,
    Vector3 ray_o, Vector3 ray_d,
    float t_min, float t_max,
    float *hit_t,
    Vector3 *hit_normal)
{
    // Set initial interval based on t_min & t_max. For a ray, tlast should be
    // set to +FLT_MAX. For a line, tfirst should also be set to –FLT_MAX
    float tfirst = t_min;
    float tlast = t_max;

    // Intersect segment against each plane
    const CountT num_faces = convex_mesh.numFaces;

    // Our face normals point outside. RTCD uses plane normals pointing inside
    // the polyhedron, so signs are flipped relative to the book
    
    Vector3 closest_normal = Vector3::zero(); 

    for (CountT face_idx = 0; face_idx < num_faces; face_idx++) {
        Plane plane = convex_mesh.facePlanes[face_idx];

        float denom = dot(plane.normal, ray_d);
        float neg_dist = plane.d - dot(plane.normal, ray_o);

        // Test if segment runs parallel to the plane
        if (denom == 0.0f) {
            // If so, return “no intersection” if segment lies outside plane
            if (neg_dist < 0.0f) return false;
        } else {
            // Compute parameterized t value for intersection with current plane
            float t = neg_dist / denom;
            if (denom < 0.0f) {
                // When entering halfspace, update tfirst if t is larger
                if (t >= tfirst) {
                    tfirst = t;
                    closest_normal = plane.normal;
                }
            } else {
                // When exiting halfspace, update tlast if t is smaller
                if (t <= tlast) {
                    tlast = t;
                }
            }
            // Exit with “no intersection” if intersection becomes empty
            if (tfirst > tlast) return false;
        }
    }

    // Addition from RTCD algo: if ray only hits backfacing planes
    // we don't set closest_normal and treat this as a miss
    if (closest_normal.x == 0 && closest_normal.y == 0 &&
        closest_normal.z == 0) {
        return false;
    }

    *hit_t = tfirst;
    *hit_normal = closest_normal;

    return true;
}

bool BVH::traceRayIntoLeaf(int32_t leaf_idx,
                           math::Vector3 world_ray_o,
                           math::Vector3 world_ray_d,
                           float t_min,
                           float t_max,
                           float *hit_t,
                           math::Vector3 *hit_normal)
{
    ObjectID obj_id = leaf_obj_ids_[leaf_idx];
    LeafTransform leaf_txfm = leaf_transforms_[leaf_idx];

    Quat rot_to_local = leaf_txfm.rot.inv();

    Vector3 obj_ray_o = rot_to_local.rotateVec(world_ray_o - leaf_txfm.pos);
    obj_ray_o.x /= leaf_txfm.scale.d0;
    obj_ray_o.y /= leaf_txfm.scale.d1;
    obj_ray_o.z /= leaf_txfm.scale.d2;

    Vector3 obj_ray_d = leaf_txfm.rot.inv().rotateVec(world_ray_d);
    obj_ray_d.x /= leaf_txfm.scale.d0;
    obj_ray_d.y /= leaf_txfm.scale.d1;
    obj_ray_d.z /= leaf_txfm.scale.d2;

    auto inv_obj_ray_d = Diag3x3::fromVec(1.f / obj_ray_d);

    Vector3 obj_hit_normal;

    CountT prim_offset = obj_mgr_->rigidBodyPrimitiveOffsets[obj_id.idx];
    CountT num_prims = obj_mgr_->rigidBodyPrimitiveCounts[obj_id.idx];

    bool hit_leaf = false;
    for (CountT i = 0; i < (CountT)num_prims; i++) {
        CountT prim_idx = prim_offset + i;

        AABB prim_aabb = obj_mgr_->primitiveAABBs[prim_idx];
        if (!prim_aabb.rayIntersects(obj_ray_o, inv_obj_ray_d, 0.f, t_max)) {
            continue;
        }

        bool hit_prim;

        const CollisionPrimitive *prim =
            &obj_mgr_->collisionPrimitives[prim_idx];
        switch (prim->type) {
        case CollisionPrimitive::Type::Hull: {
            hit_prim = traceRayIntoConvexPolyhedron(prim->hull.halfEdgeMesh,
                obj_ray_o, obj_ray_d, t_min, t_max, hit_t, &obj_hit_normal);
        } break;
        case CollisionPrimitive::Type::Plane: {
            hit_prim = traceRayIntoPlane(
                obj_ray_o, obj_ray_d, t_min, t_max, hit_t, &obj_hit_normal);
        } break;
        case CollisionPrimitive::Type::Sphere: {
            assert(false);
        } break;
        default: MADRONA_UNREACHABLE();
        }

        if (hit_prim) {
            hit_leaf = true;
            t_max = *hit_t;
        }
    }

    if (hit_leaf) {
        *hit_normal = leaf_txfm.rot.rotateVec(obj_hit_normal);

        return true;
    } else {
        return false;
    }
}

inline void updateLeafPositionsEntry(
    Context &ctx,
    const LeafID &leaf_id,
    const Position &pos,
    const Rotation &rot,
    const Scale &scale,
    const ObjectID &obj_id,
    const Velocity &vel)
{
    BVH &bvh = ctx.singleton<BVH>();
    ObjectManager &obj_mgr = *ctx.singleton<ObjectData>().mgr;
    AABB obj_aabb = obj_mgr.rigidBodyAABBs[obj_id.idx];

    bvh.updateLeafPosition(leaf_id, pos, rot, scale, vel.linear, obj_aabb);
}

// FIXME currently unused
inline void expandLeavesEntry(
    Context &ctx,
    const LeafID &leaf_id,
    const Velocity &vel)
{
    BVH &bvh = ctx.singleton<BVH>();
    AABB expanded = bvh.expandLeaf(leaf_id, vel.linear);
    bvh.refitLeaf(leaf_id, expanded);
}

inline void updateBVHEntry(Context &, BVH &bvh)
{
    bvh.updateTree();
}

inline void refitEntry(Context &ctx, LeafID leaf_id)
{
    BVH &bvh = ctx.singleton<BVH>();
    bvh.refitLeaf(leaf_id, bvh.getLeafAABB(leaf_id));
}

inline void findIntersectingEntry(
    Context &ctx,
    const Entity &e,
    LeafID leaf_id)
{
    BVH &bvh = ctx.singleton<BVH>();
    ObjectManager &obj_mgr = *ctx.singleton<ObjectData>().mgr;

    // FIXME: should have a flag for passing this
    // directly into the system
    Loc a_loc = ctx.loc(e);
    bool a_is_static = 
        ctx.getDirect<ResponseType>(RGDCols::ResponseType, a_loc) ==
        ResponseType::Static;

    ObjectID a_obj = ctx.getDirect<ObjectID>(RGDCols::ObjectID, a_loc);

    CountT a_num_prims = obj_mgr.rigidBodyPrimitiveCounts[a_obj.idx];

    bvh.findLeafIntersecting(leaf_id, [&](Entity intersecting_entity) {
        if (e.id < intersecting_entity.id) {
            Loc b_loc = ctx.loc(intersecting_entity);

            // FIXME: Change this so static objects are kept in a separate BVH
            // and this check can be removed.
            if (a_is_static &&
                ctx.getDirect<ResponseType>(RGDCols::ResponseType, b_loc) ==
                    ResponseType::Static) {
                return;
            }

            // We don't expand the primitive AABBs by movement (only object
            // AABBs) so we just unconditionally emit narrowphase checks
            // between each pair of primitives in the entity. Narrowphase
            // will check transformed AABBs.
            
            ObjectID b_obj = ctx.getDirect<ObjectID>(RGDCols::ObjectID, b_loc);
            CountT b_num_prims =
                obj_mgr.rigidBodyPrimitiveCounts[b_obj.idx];

            // FIXME: would be nice to be able to make N temporaries all at
            // once
            
            CountT total_narrowphase_checks = a_num_prims * b_num_prims;

            for (CountT prim_check_idx = 0;
                 prim_check_idx < total_narrowphase_checks;
                 prim_check_idx++) {
                CountT a_prim_idx = prim_check_idx / b_num_prims;
                CountT b_prim_idx = prim_check_idx % b_num_prims;

                Loc candidate_loc = ctx.makeTemporary<CandidateTemporary>();
                CandidateCollision &candidate =
                    ctx.getDirect<CandidateCollision>(
                        RGDCols::CandidateCollision, candidate_loc);

                candidate.a = a_loc;
                candidate.b = b_loc;
                candidate.aPrim = a_prim_idx;
                candidate.bPrim = b_prim_idx;
            }
        }
    });
}

TaskGraphNodeID setupBVHTasks(
    TaskGraphBuilder &builder,
    Span<const TaskGraphNodeID> deps)
{
    auto update_leaves =
        builder.addToGraph<ParallelForNode<Context, updateLeafPositionsEntry,
            LeafID, 
            Position,
            Rotation,
            Scale,
            ObjectID,
            Velocity>>(deps);

    auto bvh_update = builder.addToGraph<ParallelForNode<Context,
        broadphase::updateBVHEntry, broadphase::BVH>>({update_leaves});

    // FIXME Unfortunately need to call refit here, because update
    // won't necessarily do anything
    auto refit = builder.addToGraph<ParallelForNode<Context,
        broadphase::refitEntry, broadphase::LeafID>>({bvh_update});

    return refit;
}

TaskGraphNodeID setupPreIntegrationTasks(
    TaskGraphBuilder &builder,
    Span<const TaskGraphNodeID> deps)
{
    auto find_intersects = builder.addToGraph<ParallelForNode<Context,
        broadphase::findIntersectingEntry, Entity, LeafID>>(deps);

    return find_intersects;
}

TaskGraphNodeID setupPostIntegrationTasks(
    TaskGraphBuilder &builder,
    Span<const TaskGraphNodeID> deps)
{
#if 0
    auto expand_leaves = builder.addToGraph<ParallelForNode<Context,
        expandLeavesEntry, LeafID, Velocity>>({deps});
#endif

    // FIXME: can we avoid doing a full tree refit here?
    auto update_leaves =
        builder.addToGraph<ParallelForNode<Context, updateLeafPositionsEntry,
            LeafID, 
            Position,
            Rotation,
            Scale,
            ObjectID,
            Velocity>>(deps);

    auto refit = builder.addToGraph<ParallelForNode<Context,
        broadphase::refitEntry, broadphase::LeafID>>({update_leaves});

    return refit;
}

}
