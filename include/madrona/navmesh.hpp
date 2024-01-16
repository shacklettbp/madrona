#pragma once

#include <madrona/math.hpp>
#include <madrona/rand.hpp>
#include <madrona/utils.hpp>

namespace madrona {

struct Navmesh {
    using TraversalQueue = FixedSizeQueue<uint32_t>;

    struct AliasEntry {
        float tau;
        uint32_t alias;
    };

    math::Vector3 *vertices;
    uint32_t *triIndices;
    uint32_t *triAdjacency;
    AliasEntry *triSampleAliasTable;
    uint32_t numVerts;
    uint32_t numTris;

    inline math::Vector3 samplePointAndPoly(RandKey rnd, uint32_t *out_poly);
    inline math::Vector3 samplePoint(RandKey rnd);

    inline void getTriangleVertices(uint32_t tri_idx,
                                    math::Vector3 *out_a,
                                    math::Vector3 *out_b,
                                    math::Vector3 *out_c);

    template <typename Fn>
    inline void bfsFromPoly(
        uint32_t poly,
        TraversalQueue traversal_queue,
        bool *visited_tmp,
        CountT max_visited,
        Fn &&fn);

    static Navmesh initFromPolygons(
        math::Vector3 *poly_vertices,
        uint32_t *poly_idxs,
        uint32_t *poly_idx_offsets,
        uint32_t *poly_sizes,
        uint32_t num_verts,
        uint32_t num_polys);

    static constexpr inline uint32_t sentinel = 0xFFFF'FFFF;
};

}

#include "navmesh.inl"
