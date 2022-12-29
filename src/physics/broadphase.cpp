#include <madrona/context.hpp>
#include <madrona/physics.hpp>

namespace madrona {

using namespace base;
using namespace math;

namespace phys::broadphase {

BVH::BVH(CountT max_leaves)
    : nodes_((Node *)rawAlloc(sizeof(Node) *
                            numInternalNodes(max_leaves))),
      num_nodes_(0),
      num_allocated_nodes_(numInternalNodes(max_leaves)),
      leaf_aabbs_((AABB *)rawAlloc(sizeof(AABB) * max_leaves)),
      leaf_centers_((Vector3 *)rawAlloc(sizeof(Vector3) * max_leaves)),
      leaf_parents_((uint32_t *)rawAlloc(sizeof(uint32_t) * max_leaves)),
      leaf_entities_((Entity *)rawAlloc(sizeof(Entity) * max_leaves)),
      sorted_leaves_((int32_t *)rawAlloc(sizeof(int32_t) * max_leaves)),
      num_leaves_(0),
      num_allocated_leaves_(max_leaves),
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
                    return leaf_centers_[sorted_leaves_[base + offset]];
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
                     const CollisionAABB &obj_aabb)
{
    // FIXME, handle difference between potentially inflated leaf AABB and
    // object AABB
    AABB &leaf_aabb = leaf_aabbs_[leaf_id.id];
    leaf_aabb = obj_aabb;

    Vector3 &leaf_center = leaf_centers_[leaf_id.id];
    leaf_center = (leaf_aabb.pMin + leaf_aabb.pMax) / 2;

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

void updateLeavesEntry(
    Context &ctx,
    const LeafID &leaf_id,
    const CollisionAABB &aabb)
{
    BVH &bvh = ctx.getSingleton<BVH>();
    bvh.updateLeaf(leaf_id, aabb);
}

void updateBVHEntry(
    Context &, BVH &bvh)
{
    bvh.updateTree();
}

void findOverlappingEntry(
    Context &ctx,
    const Entity &e,
    const CollisionAABB &obj_aabb,
    const Velocity &)
{
    BVH &bvh = ctx.getSingleton<BVH>();

    bvh.findOverlaps(obj_aabb, [&](Entity overlapping_entity) {
        if (e.id < overlapping_entity.id) {
            Loc candidate_loc = ctx.makeTemporary<CandidateTemporary>();
            CandidateCollision &candidate = ctx.getUnsafe<
                CandidateCollision>(candidate_loc);

            candidate.a = e;
            candidate.b = overlapping_entity;
        }
    });
}

}
}

