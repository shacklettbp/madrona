#include <madrona/context.hpp>
#include <madrona/physics.hpp>

#include "physics_impl.hpp"

namespace madrona::phys::broadphase {

using namespace base;
using namespace math;

BVH::BVH(CountT max_leaves,
         float leaf_velocity_expansion,
         float leaf_accel_expansion)
    : nodes_((Node *)rawAlloc(sizeof(Node) *
                            numInternalNodes(max_leaves))),
      num_nodes_(0),
      num_allocated_nodes_(numInternalNodes(max_leaves)),
      leaf_entities_((Entity *)rawAlloc(sizeof(Entity) * max_leaves)),
      leaf_primitives_((CollisionPrimitive **)
                       rawAlloc(sizeof(CollisionPrimitive *) * max_leaves)),
      leaf_aabbs_((AABB *)rawAlloc(sizeof(AABB) * max_leaves)),
      leaf_transforms_(
          (LeafTransform *)rawAlloc(sizeof(LeafTransform) * max_leaves)),
      leaf_parents_((uint32_t *)rawAlloc(sizeof(uint32_t) * max_leaves)),
      sorted_leaves_((int32_t *)rawAlloc(sizeof(int32_t) * max_leaves)),
      num_leaves_(0),
      num_allocated_leaves_(max_leaves),
      leaf_velocity_expansion_(leaf_velocity_expansion),
      leaf_accel_expansion_(leaf_accel_expansion),
      force_rebuild_(false)
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
    int32_t num_internal_nodes =
        numInternalNodes(num_leaves_.load(std::memory_order_relaxed));
    num_nodes_ = num_internal_nodes;
    assert(num_nodes_ <= num_allocated_nodes_);

    struct StackEntry {
        int32_t nodeID;
        int32_t parentID;
        int32_t offset;
        int32_t numObjs;
    };

    StackEntry stack[128];
    stack[0] = StackEntry {
        sentinel_,
        sentinel_,
        0,
        int32_t(num_leaves_),
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
                    leaf_parents_[leaf_id] = ((uint32_t)node_id << 2) | (uint32_t)i;

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
    // validate tree AABBs
    int32_t num_leaves = num_leaves_.load(std::memory_order_relaxed);
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
#endif
}

void BVH::refit(LeafID *moved_leaf_ids, CountT num_moved)
{
    (void)moved_leaf_ids;
    (void)num_moved;

    int32_t num_moved_hacked = num_leaves_.load(std::memory_order_relaxed);

    for (CountT i = 0; i < num_moved_hacked; i++) {
        int32_t leaf_id = i;
        const AABB &leaf_aabb = leaf_aabbs_[leaf_id];
        uint32_t leaf_parent = leaf_parents_[leaf_id];

        int32_t node_idx = int32_t(leaf_parent >> 2_u32);
        int32_t sub_idx = int32_t(leaf_parent & 3);

        Node &leaf_node = nodes_[node_idx];
        leaf_node.minX[sub_idx] = leaf_aabb.pMin.x;
        leaf_node.minY[sub_idx] = leaf_aabb.pMin.y;
        leaf_node.minZ[sub_idx] = leaf_aabb.pMin.z;
        leaf_node.maxX[sub_idx] = leaf_aabb.pMax.x;
        leaf_node.maxY[sub_idx] = leaf_aabb.pMax.y;
        leaf_node.maxZ[sub_idx] = leaf_aabb.pMax.z;

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

            bool expanded = false;
            if (leaf_aabb.pMin.x < node.minX[child_offset]) {
                node.minX[child_offset] = leaf_aabb.pMin.x;
                expanded = true;
            }

            if (leaf_aabb.pMin.y < node.minY[child_offset]) {
                node.minY[child_offset] = leaf_aabb.pMin.y;
                expanded = true;
            }

            if (leaf_aabb.pMin.z < node.minZ[child_offset]) {
                node.minZ[child_offset] = leaf_aabb.pMin.z;
                expanded = true;
            }

            if (leaf_aabb.pMax.x > node.maxX[child_offset]) {
                node.maxX[child_offset] = leaf_aabb.pMax.x;
                expanded = true;
            }

            if (leaf_aabb.pMax.y > node.maxY[child_offset]) {
                node.maxY[child_offset] = leaf_aabb.pMax.y;
                expanded = true;
            }

            if (leaf_aabb.pMax.z > node.maxZ[child_offset]) {
                node.maxZ[child_offset] = leaf_aabb.pMax.z;
                expanded = true;
            }

            if (!expanded) {
                break;
            }

            child_idx = node_idx;
            node_idx = node.parentID;
        }
    }
}

void BVH::updateLeaf(LeafID leaf_id,
                     const Vector3 &pos,
                     const Quat &rot,
                     const Vector3 &scale,
                     const Vector3 &linear_vel,
                     const AABB &obj_aabb)
{
    // FIXME: this could all be more efficient with a center + width
    // AABB representation
    Mat3x3 rot_mat = Mat3x3::fromRS(rot, scale);

    // RTCD page 86
    AABB world_aabb;
#pragma unroll
    for (CountT i = 0; i < 3; i++) {
        world_aabb.pMin[i] = world_aabb.pMax[i] = pos[i];

#pragma unroll
        for (CountT j = 0; j < 3; j++) {
            float e = rot_mat[i][j] * obj_aabb.pMin[j];
            float f = rot_mat[i][j] * obj_aabb.pMax[j];

            if (e < f) {
                world_aabb.pMin[i] += e;
                world_aabb.pMax[i] += f;
            } else {
                world_aabb.pMin[i] += f;
                world_aabb.pMax[i] += e;
            }
        }
    }

#pragma unroll
    for (int32_t i = 0; i < 3; i++) {
        float pos_delta =
            leaf_velocity_expansion_ * linear_vel[i];

        float min_delta = pos_delta - leaf_accel_expansion_;
        float max_delta = pos_delta + leaf_accel_expansion_;

        if (min_delta < 0.f) {
            world_aabb.pMin[i] += min_delta;
        }
        if (max_delta > 0.f) {
            world_aabb.pMax[i] += max_delta;
        }
    }

    leaf_aabbs_[leaf_id.id] = world_aabb;
    leaf_transforms_[leaf_id.id] = {
        pos,
        rot,
        scale,
    };
    sorted_leaves_[leaf_id.id] = leaf_id.id;
}

void BVH::updateTree()
{
    if (force_rebuild_) {
        force_rebuild_ = false;
        rebuild();
    } else {
        refit(nullptr, 0);
    }
}

Entity BVH::traceRay(math::Vector3 o, math::Vector3 d, float *hit_t,
                     float t_max)
{
    using namespace math;

    Vector3 inv_d = 1.f / d;

    int32_t stack[128];
    stack[0] = 0;
    CountT stack_size = 1;

    Entity hit_entity = Entity::none();

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
                    float hit_t = traceRayIntoLeaf(leaf_idx, o, d, 0.f, t_max);

                    if (hit_t < t_max) {
                        t_max = hit_t;
                        hit_entity = leaf_entities_[leaf_idx];
                    }
                } else {
                    stack[stack_size++] = node.children[i];
                }
            }
        }
    }
    
    *hit_t = t_max;
    return hit_entity;
}

static inline float traceRayIntoPlane(
    Vector3 ray_o, Vector3 ray_d,
    float t_min, float t_max)
{
    // ray_o and ray_d have already been transformed into the space of the
    // plane. normal is (0, 0, 1), d is 0

    float denom = ray_d.z;

    if (denom == 0) {
        return t_max;
    }

    float t = -ray_o.z / denom;

    if (t >= t_min && t < t_max) {
        return t;
    } else {
        return t_max;
    }
}

// RTCD 5.3.8 (modified from segment to ray). Algorithm also in GPU Gems 2.
// Intersect ray r(t)=ray_o + , t_min <= t <=t_max against convex polyhedron
// specified by the n halfspaces defined by the planes p[]. On exit tfirst
// and tlast define the intersection, if any
static inline float traceRayIntoConvexPolyhedron(
    const geometry::HalfEdgeMesh &convex_mesh,
    Vector3 ray_o, Vector3 ray_d,
    float t_min, float t_max)
{
    // Set initial interval based on t_min & t_max. For a ray, tlast should be
    // set to +FLT_MAX. For a line, tfirst should also be set to –FLT_MAX
    float tfirst = t_min;
    float tlast = t_max;
   
    // Intersect segment against each plane
    CountT num_faces = convex_mesh.getPolygonCount();
    for (CountT face_idx = 0; face_idx < num_faces; face_idx++) {
        geometry::Plane plane = convex_mesh.getPlane(
            geometry::PolygonID(face_idx), convex_mesh.vertices());

        float denom = dot(plane.normal, ray_d);
        float dist = plane.d - dot(plane.normal, ray_o);

        // Test if segment runs parallel to the plane
        if (denom == 0.0f) {
            // If so, return “no intersection” if segment lies outside plane
            if (dist > 0.0f) return t_max;
        } else {
            // Compute parameterized t value for intersection with current plane
            float t = dist / denom;
            if (denom < 0.0f) {
                // When entering halfspace, update tfirst if t is larger
                if (t > tfirst) tfirst = t;
            } else {
                // When exiting halfspace, update tlast if t is smaller
                if (t < tlast) tlast = t;
            }
            // Exit with “no intersection” if intersection becomes empty
            if (tfirst > tlast) return t_max;
        }
    }

    return tfirst;
}

float BVH::traceRayIntoLeaf(int32_t leaf_idx,
                            math::Vector3 world_ray_o,
                            math::Vector3 world_ray_d,
                            float t_min,
                            float t_max)
{
    CollisionPrimitive *prim = leaf_primitives_[leaf_idx];
    LeafTransform leaf_txfm = leaf_transforms_[leaf_idx];

    Quat rot_to_local = leaf_txfm.rot.inv();

    Vector3 obj_ray_o = rot_to_local.rotateVec(world_ray_o - leaf_txfm.pos);
    obj_ray_o.x /= leaf_txfm.scale.x;
    obj_ray_o.y /= leaf_txfm.scale.y;
    obj_ray_o.z /= leaf_txfm.scale.z;

    Vector3 obj_ray_d = leaf_txfm.rot.inv().rotateVec(world_ray_d);
    obj_ray_d.x /= leaf_txfm.scale.x;
    obj_ray_d.y /= leaf_txfm.scale.y;
    obj_ray_d.z /= leaf_txfm.scale.z;

    switch (prim->type) {
    case CollisionPrimitive::Type::Hull: {
        return traceRayIntoConvexPolyhedron(prim->hull.halfEdgeMesh,
            obj_ray_o, obj_ray_d, t_min, t_max);
    } break;
    case CollisionPrimitive::Type::Plane: {
        return traceRayIntoPlane(obj_ray_o, obj_ray_d, t_min, t_max);
    } break;
    case CollisionPrimitive::Type::Sphere: {
        assert(false);
    } break;
    default: __builtin_unreachable();
    }

    assert(false);
}

inline void updateLeavesEntry(
    Context &ctx,
    const LeafID &leaf_id,
    const Position &pos,
    const Rotation &rot,
    const Scale &scale,
    const ObjectID &obj_id,
    const Velocity &vel)
{
    BVH &bvh = ctx.getSingleton<BVH>();
    ObjectManager &obj_mgr = *ctx.getSingleton<ObjectData>().mgr;
    AABB obj_aabb = obj_mgr.aabbs[obj_id.idx];

    bvh.updateLeaf(leaf_id, pos, rot, scale, vel.linear, obj_aabb);
}

inline void updateBVHEntry(
    Context &, BVH &bvh)
{
    bvh.updateTree();
}

inline void findOverlappingEntry(
    Context &ctx,
    const Entity &e,
    LeafID leaf_id)
{
    BVH &bvh = ctx.getSingleton<BVH>();

    bvh.findOverlapsForLeaf(leaf_id, [&](Entity overlapping_entity) {
        if (e.id < overlapping_entity.id) {
            Loc candidate_loc = ctx.makeTemporary<CandidateTemporary>();
            CandidateCollision &candidate = ctx.getUnsafe<
                CandidateCollision>(candidate_loc);

            candidate.a = e;
            candidate.b = overlapping_entity;
        }
    });
}

TaskGraph::NodeID setupTasks(
    TaskGraph::Builder &builder,
    Span<const TaskGraph::NodeID> deps)
{
    auto update_leaves = builder.addToGraph<ParallelForNode<Context,
        broadphase::updateLeavesEntry,
            LeafID, 
            Position,
            Rotation,
            Scale,
            ObjectID,
            Velocity>>(deps);

    auto bvh_update = builder.addToGraph<ParallelForNode<Context,
        broadphase::updateBVHEntry, broadphase::BVH>>({update_leaves});

    auto find_overlapping = builder.addToGraph<ParallelForNode<Context,
        broadphase::findOverlappingEntry, Entity, LeafID>>(
            {bvh_update});

    return find_overlapping;
}

}
