#include <madrona/physics.hpp>

namespace madrona {
namespace phys {

namespace {



}

BroadphaseBVH::BroadphaseBVH(CountT max_nodes)
    : nodes_((Node *)malloc(sizeof(Node) * max_nodes)),
      num_nodes_(0),
      num_allocated_nodes_(max_nodes)
{
}

void BroadphaseBVH::build(Context &ctx, Span<Entity> added_entities)
{
    int32_t num_internal_nodes =
        utils::divideRoundUp(added_entities.size() - 1, CountT(3));

    int32_t cur_node_offset = num_nodes_;
    assert(cur_node_offset == 0); // FIXME
    num_nodes_ += num_internal_nodes;
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
        int32_t(added_entities.size()),
    };

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
                    uint32_t raw_id = added_entities[entry.offset + i].id;
                    SphereObject &obj = sim.sphereObjects[obj_id];
                    node.setLeaf(i, obj_id);
                    node.minX[i] = obj.aabb.pMin.x;
                    node.minY[i] = obj.aabb.pMin.y;
                    node.minZ[i] = obj.aabb.pMin.z;
                    node.maxX[i] = obj.aabb.pMax.x;
                    node.maxY[i] = obj.aabb.pMax.y;
                    node.maxZ[i] = obj.aabb.pMax.z;
                } else {
                    node.children[i] = sentinel;
                    node.minX[i] = FLT_MAX;
                    node.minY[i] = FLT_MAX;
                    node.minZ[i] = FLT_MAX;
                    node.maxX[i] = FLT_MIN;
                    node.maxY[i] = FLT_MIN;
                    node.maxZ[i] = FLT_MIN;
                }
            }
        } else if (entry.nodeID == sentinel) {
            node_id = cur_node_offset++;
            // Record the node id in the stack entry for when this entry
            // is reprocessed
            entry.nodeID = node_id;

            BroadphaseBVHNode &node = nodes[node_id];
            for (int i = 0; i < 4; i++) {
                node.children[i] = sentinel;
            }
            node.parentID = entry.parentID;

            // midpoint sort items
            auto midpoint_split = [&sim, added_objects](
                                      int32_t base, int32_t num_elems) {
                Vector3 center_min {
                    FLT_MAX,
                    FLT_MAX,
                    FLT_MAX,
                };

                Vector3 center_max {
                    FLT_MIN,
                    FLT_MIN,
                    FLT_MIN,
                };

                for (int i = 0; i < num_elems; i++) {
                    int32_t obj_id = added_objects[base + i];
                    SphereObject &obj = sim.sphereObjects[obj_id];
                    center_min = Vector3::min(center_min, obj.physCenter);
                    center_max = Vector3::max(center_max, obj.physCenter);
                }

                auto split = [&](auto get_component) {
                    float split_val = 0.5f * (get_component(center_min) +
                                              get_component(center_max));

                    int start = 0;
                    int end = num_elems;

                    while (start < end) {
                        auto center_component = [&](int32_t idx) {
                            int32_t obj_id = added_objects[base + idx];
                            return get_component(
                                sim.sphereObjects[obj_id].physCenter);
                        };

                        while (start < end && center_component(start) < 
                               split_val) {
                            ++start;
                        }

                        while (start < end && center_component(end - 1) >=
                               split_val) {
                            --end;
                        }

                        if (start < end) {
                            std::swap(added_objects[base + start],
                                      added_objects[base + end - 1]);
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

#if 0
            printf("%u %u\n", entry.offset, entry.numObjs);
            printf("[%u %u) [%u %u) [%u %u) [%u %u)\n",
                   entry.offset, entry.offset + first_split,
                   entry.offset + first_split, entry.offset + first_split + num_h1 - first_split,
                   entry.offset + num_h1, entry.offset + num_h1 + third_split,
                   entry.offset + num_h1 + third_split, entry.offset + num_h1 + third_split + num_h2 - third_split);
#endif

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

        BroadphaseBVHNode &node = nodes[node_id];
        if (node.parentID == -1) {
            continue;
        }

        AABB combined_aabb  = AABB::invalid(); 
        for (int i = 0; i < 4; i++) {
            if (node.children[i] == sentinel) {
                break;
            }

            combined_aabb = AABB::merge(combined_aabb, AABB {
                .pMin = {
                    node.minX[i],
                    node.minY[i],
                    node.minZ[i],
                },
                .pMax = {
                    node.maxX[i],
                    node.maxY[i],
                    node.maxZ[i],
                },
            });
        }

        BroadphaseBVHNode &parent = nodes[node.parentID];
        int child_offset;
        for (child_offset = 0; ; child_offset++) {
            if (parent.children[child_offset] == sentinel) {
                break;
            }
        }

        parent.children[child_offset] = entry.nodeID;
        parent.minX[child_offset] = combined_aabb.pMin.x;
        parent.minY[child_offset] = combined_aabb.pMin.y;
        parent.minZ[child_offset] = combined_aabb.pMin.z;
        parent.maxX[child_offset] = combined_aabb.pMax.x;
        parent.maxY[child_offset] = combined_aabb.pMax.y;
        parent.maxZ[child_offset] = combined_aabb.pMax.z;
    }
}

void BroadphaseBVH::update(SimpleSim &sim,
                        int32_t *added_objects, int32_t num_added_objects,
                        int32_t *removed_objects, int32_t num_removed_objects,
                        int32_t *moved_objects, int32_t num_moved_objects)
{
    if (num_added_objects > 0) {
        build(sim, added_objects, num_added_objects);
    }

    (void)removed_objects;
    (void)num_removed_objects;

    for (int i = 0; i < (int)num_moved_objects; i++) {
        int32_t obj_id = moved_objects[i];
        SphereObject &obj = sim.sphereObjects[obj_id];
        AABB obj_aabb = obj.aabb;

        int32_t node_idx = int32_t(obj.leafID >> 2_u32);
        int32_t sub_idx = int32_t(obj.leafID & 3);

        BroadphaseBVHNode &leaf_node = nodes[node_idx];
        leaf_node.minX[sub_idx] = obj_aabb.pMin.x;
        leaf_node.minY[sub_idx] = obj_aabb.pMin.y;
        leaf_node.minZ[sub_idx] = obj_aabb.pMin.z;
        leaf_node.maxX[sub_idx] = obj_aabb.pMax.x;
        leaf_node.maxY[sub_idx] = obj_aabb.pMax.y;
        leaf_node.maxZ[sub_idx] = obj_aabb.pMax.z;

        int32_t child_idx = node_idx;
        node_idx = leaf_node.parentID;

        while (node_idx != sentinel) {
            BroadphaseBVHNode &node = nodes[node_idx];
            int child_offset = -1;
            for (int j = 0; j < 4; j++) {
                if (node.children[j] == child_idx) {
                    child_offset = j;
                    break;
                }
            }
            assert(child_offset != -1);

            bool expanded = false;
            if (obj_aabb.pMin.x < node.minX[child_offset]) {
                node.minX[child_offset] = obj_aabb.pMin.x;
                expanded = true;
            }

            if (obj_aabb.pMin.y < node.minY[child_offset]) {
                node.minY[child_offset] = obj_aabb.pMin.y;
                expanded = true;
            }

            if (obj_aabb.pMin.z < node.minZ[child_offset]) {
                node.minZ[child_offset] = obj_aabb.pMin.z;
                expanded = true;
            }

            if (obj_aabb.pMax.x > node.maxX[child_offset]) {
                node.maxX[child_offset] = obj_aabb.pMax.x;
                expanded = true;
            }

            if (obj_aabb.pMax.y > node.maxY[child_offset]) {
                node.maxY[child_offset] = obj_aabb.pMax.y;
                expanded = true;
            }

            if (obj_aabb.pMax.z > node.maxZ[child_offset]) {
                node.maxZ[child_offset] = obj_aabb.pMax.z;
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


}
}
