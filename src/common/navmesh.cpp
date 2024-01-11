#include <madrona/navmesh.hpp>
#include <madrona/utils.hpp>
#include <madrona/memory.hpp>

namespace madrona {

using namespace math;

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
        uint32_t poly_idx_offset = poly_idx_offsets[i];
        for (uint32_t tri_base_offset = 0; tri_base_offset < poly_size - 2;
             tri_base_offset++) {
            uint32_t idx_a = poly_idxs[poly_idx_offset + tri_base_offset];
            uint32_t idx_b = poly_idxs[poly_idx_offset + tri_base_offset + 1];
            uint32_t idx_c = poly_idxs[poly_idx_offset + tri_base_offset + 2];

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
            tri_weights[over_idx] + tri_weights[under_idx] - 1.f;
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

    return Navmesh {
        .vertices = out_vertices,
        .triIndices = tri_indices,
        .triAdjacency = nullptr,
        .triSampleAliasTable = alias_tbl,
        .numVerts = num_verts,
        .numTris = num_tris,
    };
}

}
