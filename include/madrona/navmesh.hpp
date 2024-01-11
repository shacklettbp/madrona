#pragma once

#include <madrona/math.hpp>
#include <madrona/rand.hpp>

namespace madrona {

struct Navmesh {
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

    inline math::Vector3 samplePoint(RandKey rnd);

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
