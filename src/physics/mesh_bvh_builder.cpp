#include <madrona/physics_assets.hpp>
#include <madrona/macros.hpp>

namespace madrona::phys {

using namespace madrona::math;

void * MeshBVHBuilder::build(Span<const imp::SourceMesh> src_meshes,
                             StackAlloc &tmp_alloc,
                             MeshBVH *out_bvh,
                             CountT *out_num_bytes)
{
    using Node = MeshBVH::Node;
    using LeafGeometry = MeshBVH::LeafGeometry;
    using LeafMaterial = MeshBVH::LeafMaterial;

    int32_t total_num_verts = 0;
    int32_t total_num_tris = 0;
    for (const imp::SourceMesh &src_mesh : src_meshes) {
        if (src_mesh.faceCounts != nullptr) {
            FATAL("MeshBVH only supports triangular meshes");
        }

        total_num_verts += (int32_t)src_mesh.numVertices;
        total_num_tris += (int32_t)src_mesh.numFaces;
    }
    
    auto tmp_frame = tmp_alloc.push();

    Vector3 *combined_verts = tmp_alloc.allocN<Vector3>(total_num_verts);
    uint64_t *combined_tri_indices =
        tmp_alloc.allocN<uint64_t>(total_num_tris);
    uint32_t *combined_tri_mats =
        tmp_alloc.allocN<uint32_t>(total_num_tris);
    AABB *tri_aabbs = tmp_alloc.allocN<AABB>(total_num_tris);

    uint32_t *tri_reorder =
        tmp_alloc.allocN<uint32_t>(total_num_tris);

    int32_t cur_tri_offset = 0;
    int32_t cur_vert_offset = 0;
    for (const imp::SourceMesh &src_mesh : src_meshes) {
        memcpy(combined_verts + cur_vert_offset, src_mesh.positions,
               sizeof(Vector3) * src_mesh.numVertices);

        for (int32_t i = 0 ; i < (int32_t)src_mesh.numFaces; i++) {
            int32_t base = 3 * i;
            uint32_t mesh_a_idx = src_mesh.indices[base + 0];
            uint32_t mesh_b_idx = src_mesh.indices[base + 1];
            uint32_t mesh_c_idx = src_mesh.indices[base + 2];

            {
                uint32_t global_a_idx = mesh_a_idx + cur_vert_offset;
                uint32_t global_b_idx = mesh_b_idx + cur_vert_offset;
                uint32_t global_c_idx = mesh_c_idx + cur_vert_offset;

                int32_t b_diff = (int32_t)global_b_idx - (int32_t)global_a_idx;
                int32_t c_diff = (int32_t)global_c_idx - (int32_t)global_a_idx;
                assert(abs(b_diff) < 32767 && abs(c_diff) < 32767);

                combined_tri_indices[cur_tri_offset + i] =
                    (uint64_t(global_a_idx) << 32) |
                    (uint64_t((uint16_t)b_diff) << 16) |
                    uint64_t((uint16_t)c_diff);
            }
            combined_tri_mats[cur_tri_offset + i] =
                src_mesh.faceMaterials ? src_mesh.faceMaterials[i] :
                0xFFFF'FFFF;

            Vector3 a = src_mesh.positions[mesh_a_idx];
            Vector3 b = src_mesh.positions[mesh_b_idx];
            Vector3 c = src_mesh.positions[mesh_c_idx];

            AABB tri_aabb = AABB::point(a);
            tri_aabb.expand(b);
            tri_aabb.expand(c);

            tri_aabbs[cur_tri_offset + i] = tri_aabb;
            tri_reorder[cur_tri_offset + i] = cur_tri_offset + i;
        }

        cur_tri_offset += src_mesh.numFaces;
        cur_vert_offset += src_mesh.numVertices;
    }

    // FIXME: Neither of these bounds are tight, because the current
    // BVH build code has a problem where leaves / nodes aren't guaranteed
    // to be tightly packed.
    int32_t max_num_leaves = total_num_tris;
    int32_t max_num_nodes = std::max(utils::divideRoundUp(max_num_leaves - 1,
        (int32_t)MeshBVH::nodeWidth - 1), int32_t(1)) + max_num_leaves;

    Node *nodes = tmp_alloc.allocN<Node>(max_num_nodes);
    LeafGeometry *leaf_geos =
        tmp_alloc.allocN<LeafGeometry>(max_num_leaves);
    LeafMaterial *leaf_mats =
        tmp_alloc.allocN<LeafMaterial>(max_num_leaves);

    // midpoint sort items
    auto midpoint_split = [&](int32_t base, int32_t num_elems) {
        auto get_center = [&](int32_t offset) {
            AABB aabb = tri_aabbs[tri_reorder[base + offset]];

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
                    std::swap(tri_reorder[base + start],
                              tri_reorder[base + end - 1]);
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
        } else if (center_diff.y > center_diff.z) {
            return split([](Vector3 v) {
                return v.y;
            });
        } else {
            return split([](Vector3 v) {
                return v.z;
            });
        }
    };

    struct StackEntry {
        int32_t nodeID;
        int32_t parentID;
        int32_t offset;
        int32_t numTris;
    };

    StackEntry stack[128];
    stack[0] = StackEntry {
        MeshBVH::sentinel,
        MeshBVH::sentinel,
        0,
        (int32_t)total_num_tris,
    };

    int32_t cur_node_offset = 0;
    int32_t cur_leaf_offset = 0;

    CountT stack_size = 1;

    while (stack_size > 0) {
        StackEntry &entry = stack[stack_size - 1];
        int32_t node_id;

        if (entry.nodeID == MeshBVH::sentinel) {
            node_id = cur_node_offset++;
            assert(node_id < max_num_nodes);

            Node &node = nodes[node_id];
            for (int32_t i = 0; i < MeshBVH::nodeWidth; i++) {
                node.clearChild(i);
                node.minX[i] = FLT_MAX;
                node.minY[i] = FLT_MAX;
                node.minZ[i] = FLT_MAX;
                node.maxX[i] = -FLT_MAX;
                node.maxY[i] = -FLT_MAX;
                node.maxZ[i] = -FLT_MAX;
            }
            node.parentID = entry.parentID;

            int32_t second_split = midpoint_split(entry.offset, entry.numTris);
            int32_t num_h1 = second_split;
            int32_t num_h2 = entry.numTris - second_split;

            int32_t first_split = midpoint_split(entry.offset, num_h1);
            int32_t third_split =
                midpoint_split(entry.offset + second_split, num_h2);

            int32_t subdiv_starts[MeshBVH::nodeWidth];
            int32_t subdiv_counts[MeshBVH::nodeWidth];

            subdiv_starts[0] = entry.offset;
            subdiv_counts[0] = first_split;

            subdiv_starts[1] = entry.offset + first_split;
            subdiv_counts[1] = num_h1 - first_split;

            subdiv_starts[2] = entry.offset + num_h1;
            subdiv_counts[2] = third_split;

            subdiv_starts[3] = entry.offset + num_h1 + third_split;
            subdiv_counts[3] = num_h2 - third_split;

            bool has_non_leaf_children = false;
            // Process children in reverse order to preserve left-right
            // depth first ordering after popping off stack
            for (int32_t i = MeshBVH::nodeWidth - 1; i >= 0; i--) {
                int32_t node_tri_start = subdiv_starts[i];
                int32_t node_tri_count = subdiv_counts[i];

                if (node_tri_count == 0) {
                    continue;
                }

                if (node_tri_count > MeshBVH::numTrisPerLeaf) {
                    assert(stack_size < 128 - 1);
                    stack[stack_size++] = {
                        -1,
                        node_id,
                        node_tri_start,
                        node_tri_count,
                    };

                    has_non_leaf_children = true;
                    continue;
                }

                int32_t leaf_idx = cur_leaf_offset++;
                assert(leaf_idx < max_num_leaves);

                AABB leaf_aabb = AABB::invalid();
                for (int32_t tri_offset = 0; tri_offset < node_tri_count;
                     tri_offset++) {
                    int32_t tri_idx =
                        tri_reorder[node_tri_start + tri_offset];
                    leaf_aabb = AABB::merge(leaf_aabb, tri_aabbs[tri_idx]);

                    leaf_geos[leaf_idx].packedIndices[tri_offset] =
                        combined_tri_indices[tri_idx];

                    leaf_mats[leaf_idx].material[tri_offset] =
                        combined_tri_mats[tri_idx];
                }

                for (int32_t tri_offset = node_tri_count;
                     tri_offset < MeshBVH::numTrisPerLeaf;
                     tri_offset++) {
                    leaf_geos[leaf_idx].packedIndices[tri_offset] =
                        0xFFFF'FFFF'FFFF'FFFF;
                }

                node.setLeaf(i, leaf_idx);

                node.minX[i] = leaf_aabb.pMin.x;
                node.minY[i] = leaf_aabb.pMin.y;
                node.minZ[i] = leaf_aabb.pMin.z;
                node.maxX[i] = leaf_aabb.pMax.x;
                node.maxY[i] = leaf_aabb.pMax.y;
                node.maxZ[i] = leaf_aabb.pMax.z;
            }

            if (has_non_leaf_children) {
                // Record the node id in the stack entry for when this entry
                // is reprocessed
                entry.nodeID = node_id;

                // Defer processing this node until children are processed
                continue;
            }
        } else {
            // Revisiting this node after having processed children
            node_id = entry.nodeID;
        }

        // At this point, remove the current entry from the stack
        stack_size -= 1;

        Node &node = nodes[node_id];
        if (node.parentID == -1) {
            continue;
        }

        AABB combined_aabb = AABB::invalid();
        for (CountT i = 0; i < 4; i++) {
            if (!node.hasChild(i)) {
                continue;
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

        Node &parent = nodes[node.parentID];
        CountT child_offset;
        for (child_offset = 0; ; child_offset++) {
            if (parent.children[child_offset] == MeshBVH::sentinel) {
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

    AABB root_aabb = AABB::invalid();
    {
        const auto &root_node = nodes[0];
        for (int32_t i = 0; i < MeshBVH::nodeWidth; i++) {
            if (root_node.children[i] == MeshBVH::sentinel) {
                continue;
            }

            Vector3 p_min {
                root_node.minX[i],
                root_node.minY[i],
                root_node.minZ[i],
            };

            Vector3 p_max {
                root_node.maxX[i],
                root_node.maxY[i],
                root_node.maxZ[i],
            };

            root_aabb = AABB::merge(root_aabb, AABB {
                .pMin = p_min,
                .pMax = p_max,
            });
        }
    }

    int32_t num_nodes = cur_node_offset;
    int32_t num_leaves = cur_leaf_offset;

    auto buffer_sizes = std::to_array<int64_t>({
        (int64_t)sizeof(Node) * num_nodes,
        (int64_t)sizeof(LeafGeometry) * num_leaves,
        (int64_t)sizeof(LeafMaterial) * num_leaves,
        (int64_t)sizeof(Vector3) * total_num_verts,
    });

    int64_t buffer_offsets[buffer_sizes.size() - 1];
    int64_t total_num_bytes = utils::computeBufferOffsets(
        buffer_sizes, buffer_offsets, MADRONA_CACHE_LINE);

    char *buffer = (char *)malloc(total_num_bytes);

    *out_bvh = MeshBVH {
        .nodes = (Node * )(buffer),
        .leafGeos = (LeafGeometry *)(buffer + buffer_offsets[0]),
        .leafMats = (LeafMaterial *)(buffer + buffer_offsets[1]),
        .vertices = (Vector3 *)(buffer + buffer_offsets[2]),
        .rootAABB = root_aabb,
        .numNodes = (uint32_t)num_nodes,
        .numLeaves = (uint32_t)num_leaves,
        .numVerts = (uint32_t)total_num_verts,
    };
    *out_num_bytes = total_num_bytes;

    memcpy(out_bvh->nodes, nodes, sizeof(Node) * num_nodes);
    memcpy(out_bvh->leafGeos, leaf_geos, sizeof(LeafGeometry) * num_leaves);
    memcpy(out_bvh->leafMats, leaf_mats, sizeof(LeafMaterial) * num_leaves);
    memcpy(out_bvh->vertices, combined_verts,
           sizeof(Vector3) * total_num_verts);

    tmp_alloc.pop(tmp_frame);

    return buffer;
}

}
