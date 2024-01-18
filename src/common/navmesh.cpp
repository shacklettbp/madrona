#include <madrona/navmesh.hpp>
#include <madrona/utils.hpp>
#include <madrona/memory.hpp>

namespace madrona {

using namespace math;

static inline CountT heapParent(CountT idx)
{
    return (idx - 1) / 2;
}

static inline CountT heapChildOffset(CountT idx)
{
    return 2 * idx + 1;
}

static inline void heapMoveUp(CountT moved_idx,
                              uint32_t moved_poly,
                              float moved_cost,
                              uint32_t *heap,
                              uint32_t *heap_index,
                              float *costs)
{
    while (moved_idx != 0) {
        CountT parent_idx = heapParent(moved_idx);
        uint32_t parent_poly = heap[parent_idx];
        if (costs[parent_poly] <= moved_cost) {
            break;
        }

        heap[moved_idx] = parent_poly;
        heap_index[parent_poly] = moved_idx;

        moved_idx = parent_idx;
    }

    heap[moved_idx] = moved_poly;
    heap_index[moved_poly] = moved_idx;
}

void Navmesh::PathFindQueue::add(uint32_t poly, float cost)
{
    costs[poly] = cost;

    CountT new_idx = heapSize++;
    heapMoveUp(new_idx, poly, cost, heap, heapIndex, costs);
}

uint32_t Navmesh::PathFindQueue::removeMin()
{
    uint32_t root_poly = heap[0];

    uint32_t moved_poly = heap[--heapSize];
    float moved_cost = costs[moved_poly];

    CountT moved_idx = 0;
    CountT child_offset;
    while ((child_offset = heapChildOffset(moved_idx)) < heapSize) {
        CountT child_idx = child_offset;
        uint32_t child_poly = heap[child_idx];
        float child_cost = costs[child_poly];
        {
            // Pick the lowest cost child
            CountT right_idx = child_idx + 1;
            if (right_idx < heapSize) {
                uint32_t right_poly = heap[right_idx];
                float right_cost = costs[right_poly];
                if (right_cost < child_cost) {
                    child_idx = right_idx;
                    child_poly = right_poly;
                    child_cost = right_cost;
                }
            }
        }

        // moved_idx is now a valid position for moved_poly in the heap
        if (moved_cost < child_cost) {
            break;
        }

        heap[moved_idx] = child_poly;
        heapIndex[child_poly] = moved_idx;

        moved_idx = child_idx;
    }
        
    heap[moved_idx] = moved_poly;
    heapIndex[moved_poly] = moved_idx;

    heapIndex[root_poly] = Navmesh::sentinel;
    return root_poly;
}

void Navmesh::PathFindQueue::decreaseCost(uint32_t poly, float cost)
{
    costs[poly] = cost;

    CountT cur_idx = (CountT)heapIndex[poly];

    heapMoveUp(cur_idx, poly, cost, heap, heapIndex, costs);
}

static inline uint32_t hashNavmeshEdge(uint32_t a, uint32_t b)
{
    // MurmurHash2 Finalizer
    
    const uint32_t m = 0x5bd1e995;
    
    a ^= b >> 18;
    a *= m;
    b ^= a >> 22;
    b *= m;
    a ^= b >> 17;
    a *= m;
    b ^= a >> 19;
    b *= m;
    
    return b;
}

Navmesh Navmesh::initFromPolygons(
    Vector3 *poly_vertices,
    uint32_t *poly_idxs,
    uint32_t *poly_idx_offsets,
    uint32_t *poly_sizes,
    uint32_t num_verts,
    uint32_t num_polys)
{
    Vector3 *out_vertices = (Vector3 *)rawAlloc(sizeof(Vector3) * num_verts);
    utils::copyN<Vector3>(out_vertices, poly_vertices, num_verts);

    uint32_t num_tris = 0;
    for (CountT i = 0; i < (CountT)num_polys; i++) {
        uint32_t poly_size = poly_sizes[i];

        num_tris += poly_size - 2;
    }

    uint32_t *tri_indices =
        (uint32_t *)rawAlloc(sizeof(uint32_t) * 3 * num_tris);
    uint32_t *tri_adjacency =
        (uint32_t *)rawAlloc(sizeof(uint32_t) * 3 * num_tris);

    AliasEntry *alias_tbl =
        (AliasEntry *)rawAlloc(sizeof(AliasEntry) * num_tris);

    // Temporary data
    float *tri_weights = (float *)rawAlloc(sizeof(float) * num_tris);
    uint32_t *alias_stack =
        (uint32_t *)rawAlloc(sizeof(uint32_t) * num_tris * 2);
    uint32_t *under_stack = alias_stack;
    uint32_t *over_stack = alias_stack + num_tris;
    uint32_t under_stack_size = 0;
    uint32_t over_stack_size = 0;

    // Triangulate the input polygons
    float tri_weight_sum = 0.f;
    uint32_t cur_tri = 0;
    for (CountT i = 0; i < (CountT)num_polys; i++) {
        uint32_t poly_size = poly_sizes[i];
        uint32_t poly_idx_base = poly_idx_offsets[i];
        for (uint32_t tri_offset = 1; tri_offset < poly_size - 1; tri_offset++) {
            uint32_t idx_a = poly_idxs[poly_idx_base];
            uint32_t idx_b = poly_idxs[poly_idx_base + tri_offset];
            uint32_t idx_c = poly_idxs[poly_idx_base + tri_offset + 1];

            tri_indices[3 * cur_tri] = idx_a;
            tri_indices[3 * cur_tri + 1] = idx_b;
            tri_indices[3 * cur_tri + 2] = idx_c;

            Vector3 a = poly_vertices[idx_a];
            Vector3 b = poly_vertices[idx_b];
            Vector3 c = poly_vertices[idx_c];

            Vector3 ab = b - a;
            Vector3 ac = c - a;
            float tri_area_x2 = cross(ab, ac).length();
            tri_weights[cur_tri] = tri_area_x2;
            tri_weight_sum += tri_area_x2;

            cur_tri++;
        }
    }

    for (uint32_t tri_idx = 0; tri_idx < num_tris; tri_idx++) {
        float normalized_weight =
            tri_weights[tri_idx] * float(num_tris) / tri_weight_sum;
        tri_weights[tri_idx] = normalized_weight;

        if (normalized_weight < 1.f) {
            under_stack[under_stack_size++] = tri_idx;
        } else {
            over_stack[over_stack_size++] = tri_idx;
        }
    }

    while (under_stack_size != 0 && over_stack_size != 0) {
        uint32_t under_idx = under_stack[--under_stack_size];
        uint32_t over_idx = over_stack[--over_stack_size];

        alias_tbl[under_idx] = {
            .tau = tri_weights[under_idx],
            .alias = over_idx,
        };

        float new_over_weight =
            (tri_weights[over_idx] + tri_weights[under_idx]) - 1.f;
        tri_weights[over_idx] = new_over_weight;

        if (new_over_weight < 1.f) {
            under_stack[under_stack_size++] = over_idx;
        } else {
            over_stack[over_stack_size++] = over_idx;
        }
    }

    for (uint32_t i = 0; i < under_stack_size; i++) {
        uint32_t idx = under_stack[i];

        alias_tbl[idx] = {
            .tau = 1.f,
            .alias = idx, // never accessed
        };
    }

    for (uint32_t i = 0; i < over_stack_size; i++) {
        uint32_t idx = over_stack[i];

        alias_tbl[idx] = {
            .tau = 1.f,
            .alias = idx, // never accessed
        };
    }

    rawDealloc(alias_stack);
    rawDealloc(tri_weights);

    struct EdgeEntry {
        uint32_t vertA;
        uint32_t vertB;
        uint32_t firstTriIdx;
        uint32_t firstTriEdgeOffset;
    };

    uint32_t max_edges = num_tris * 3;
    auto *edge_tbl = (EdgeEntry *)rawAlloc(sizeof(EdgeEntry) * max_edges);

    for (uint32_t i = 0; i < max_edges; i++) {
        tri_adjacency[i] = sentinel;
        edge_tbl[i] = { sentinel, sentinel, sentinel, 0 };
    }

    // This uses a super simple open addressing hash table. The max_edges bound
    // is quite pessimistic, so shouldn't get too close to full on the hash
    // table.
    auto recordEdge = [edge_tbl, tri_adjacency, max_edges, out_vertices](
        uint32_t tri_idx,
        uint32_t tri_edge_offset,
        uint32_t a, uint32_t b)
    {
        if (b < a) {
            std::swap(a, b);
        }

        // Use Lemire fast modulo replacement trick
        uint32_t edge_hash = utils::u32mulhi(hashNavmeshEdge(a, b), max_edges);

        while (edge_tbl[edge_hash].vertA != sentinel && (
                edge_tbl[edge_hash].vertA != a || 
                edge_tbl[edge_hash].vertB != b)) {
            if (edge_hash == max_edges - 1) {
                edge_hash = 0;
            } else {
                edge_hash += 1;
            }
        }

        EdgeEntry &entry = edge_tbl[edge_hash];
        entry.vertA = a;
        entry.vertB = b;

        if (entry.firstTriIdx == sentinel) {
            entry.firstTriIdx = tri_idx;
            entry.firstTriEdgeOffset = tri_edge_offset;
        } else {
            uint32_t other_tri_idx = entry.firstTriIdx;
            uint32_t other_tri_edge_offset = entry.firstTriEdgeOffset;

            tri_adjacency[3 * tri_idx + tri_edge_offset] = other_tri_idx;
            tri_adjacency[3 * other_tri_idx + other_tri_edge_offset] = tri_idx;
        }
    };

    for (uint32_t tri_idx = 0; tri_idx < num_tris; tri_idx++) {
        uint32_t a_idx = tri_indices[3 * tri_idx];
        uint32_t b_idx = tri_indices[3 * tri_idx + 1];
        uint32_t c_idx = tri_indices[3 * tri_idx + 2];

        recordEdge(tri_idx, 0, a_idx, b_idx);
        recordEdge(tri_idx, 1, b_idx, c_idx);
        recordEdge(tri_idx, 2, c_idx, a_idx);
    }

    rawDealloc(edge_tbl);

    return Navmesh {
        .vertices = out_vertices,
        .triIndices = tri_indices,
        .triAdjacency = tri_adjacency,
        .triSampleAliasTable = alias_tbl,
        .numVerts = num_verts,
        .numTris = num_tris,
    };
}

}
