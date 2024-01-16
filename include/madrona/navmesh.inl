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
                          TraversalQueue traversal_queue,
                          bool *visited_tmp,
                          CountT max_visited,
                          Fn &&fn)
{
    // FIXME
    assert(max_visited == numTris);

    utils::zeroN<bool>(visited_tmp, numTris);

    traversal_queue.add(start_poly);
    visited_tmp[start_poly] = true;

    while (!traversal_queue.isEmpty()) {
        uint32_t poly = traversal_queue.remove();

        bool accept = fn(poly);
        if (!accept) {
            continue;
        }

        MADRONA_UNROLL
        for (CountT i = 0; i < 3; i++) {
            uint32_t adjacent = triAdjacency[3 * poly + i];

            if (adjacent != sentinel && !visited_tmp[adjacent]) {
                traversal_queue.add(adjacent);
                visited_tmp[adjacent] = true;
            }
        }
    }
}

}
