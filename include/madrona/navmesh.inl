#include <cassert>

namespace madrona {

math::Vector3 Navmesh::samplePointAndPoly(RandKey rnd, uint32_t *out_poly)
{
    using namespace math;

    RandKey tbl_row_rnd = rand::split_i(rnd, 0);
    RandKey alias_p_rnd = rand::split_i(rnd, 1);
    RandKey bary_rnd = rand::split_i(rnd, 2);

    uint32_t tbl_row_idx = rand::sampleI32(tbl_row_rnd, 0, numTris);
    float p = rand::sampleUniform(alias_p_rnd);

    AliasEntry tbl_row = triSampleAliasTable[tbl_row_idx];

    uint32_t tri_idx = p < tbl_row.tau ? tbl_row_idx : tbl_row.alias;
    *out_poly = tri_idx;

    Vector3 a, b, c;
    getTriangleVertices(tri_idx, &a, &b, &c);

    Vector2 uv = rand::sample2xUniform(bary_rnd);

    if (uv.x + uv.y > 1.f) {
        uv.x = 1.f - uv.x;
        uv.y = 1.f - uv.y;
    }

    float w = 1.f - uv.x - uv.y;

    return a * uv.x + b * uv.y + c * w;
}

math::Vector3 Navmesh::samplePoint(RandKey rnd)
{
    uint32_t poly;
    return samplePointAndPoly(rnd, &poly);
}

void Navmesh::getTriangleVertices(uint32_t tri_idx,
                                  math::Vector3 *out_a,
                                  math::Vector3 *out_b,
                                  math::Vector3 *out_c)
{
    *out_a = vertices[triIndices[3 * tri_idx]];
    *out_b = vertices[triIndices[3 * tri_idx + 1]];
    *out_c = vertices[triIndices[3 * tri_idx + 2]];
}

template <typename Fn>
void Navmesh::bfsFromPoly(uint32_t start_poly,
                          BFSState bfs_state,
                          Fn &&fn)
{
    ArrayQueue<uint32_t> bfs_queue(bfs_state.queue, numTris);
    bool *visited = bfs_state.visited;

    utils::zeroN<bool>(visited, numTris);

    bfs_queue.add(start_poly);
    visited[start_poly] = true;

    while (!bfs_queue.isEmpty()) {
        uint32_t poly = bfs_queue.remove();

        bool accept = fn(poly);
        if (!accept) {
            continue;
        }

        MADRONA_UNROLL
        for (CountT i = 0; i < 3; i++) {
            uint32_t adjacent = triAdjacency[3 * poly + i];

            if (adjacent != sentinel && !visited[adjacent]) {
                bfs_queue.add(adjacent);
                visited[adjacent] = true;
            }
        }
    }
}


template <typename Fn>
inline void Navmesh::dijkstrasFromPoly(
    uint32_t start_poly,
    math::Vector3 start_pos,
    DijkstrasState dijkstras_state,
    Fn &&fn)
{
    using namespace math;

    float *distances = dijkstras_state.distances;

    PathFindQueue prio_queue {
        .costs = distances,
        .heap = dijkstras_state.heap,
        .heapIndex = dijkstras_state.heapIndex,
        .heapSize = 0,
    };
    utils::fillN<uint32_t>(prio_queue.heapIndex, sentinel, numTris);
    utils::fillN<float>(distances, FLT_MAX, numTris);

    Vector3 *entry_points = dijkstras_state.entryPoints;
    entry_points[start_poly] = start_pos;

    prio_queue.add(start_poly, 0.f);
    while (prio_queue.heapSize > 0) {
        uint32_t min_poly = prio_queue.removeMin();
        Vector3 cur_pos = entry_points[min_poly];
        float dist_so_far = distances[min_poly];

        fn(min_poly, cur_pos, dist_so_far);
        
        Vector3 edge_midpoints[3];
        {
            Vector3 a, b, c;
            getTriangleVertices(min_poly, &a, &b, &c);

            edge_midpoints[0] = (a + b) / 2.f;
            edge_midpoints[1] = (b + c) / 2.f;
            edge_midpoints[2] = (c + a) / 2.f;
        }

        MADRONA_UNROLL
        for (CountT i = 0; i < 3; i++) {
            uint32_t adjacent = triAdjacency[3 * min_poly + i];
            if (adjacent == Navmesh::sentinel) {
                continue;
            }

            Vector3 edge_midpoint = edge_midpoints[i];

            float dist_to_edge = cur_pos.distance(edge_midpoint);
            float new_dist = dist_so_far + dist_to_edge;
            float prev_dist = distances[adjacent];

            if (new_dist >= prev_dist) {
                continue;
            }

            entry_points[adjacent] = edge_midpoint;

            uint32_t prio_queue_idx = prio_queue.heapIndex[adjacent];
            if (prio_queue_idx == sentinel) {
                prio_queue.add(adjacent, new_dist);
            } else {
                prio_queue.decreaseCost(adjacent, new_dist);
            }
        }
    }
}

}
