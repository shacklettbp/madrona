#pragma once

#include <madrona/math.hpp>
#include <madrona/rand.hpp>
#include <madrona/utils.hpp>

namespace madrona {

struct Navmesh {
    struct PathFindQueue {
        float *costs;
        uint32_t *heap;
        uint32_t *heapIndex;
        CountT heapSize;

        void add(uint32_t poly, float cost);
        uint32_t removeMin();
        void decreaseCost(uint32_t poly, float cost);
    };

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

    struct BFSState {
        uint32_t *queue;
        bool *visited;
    };

    template <typename Fn>
    inline void bfsFromPoly(
        uint32_t poly,
        BFSState bfs_state,
        Fn &&fn);

    struct DijkstrasState {
        float *distances;
        math::Vector3 *entryPoints;
        uint32_t *heap;
        uint32_t *heapIndex;
    };

    template <typename Fn>
    inline void dijkstrasFromPoly(
        uint32_t start_poly,
        math::Vector3 start_pos,
        DijkstrasState dijkstras_state,
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
