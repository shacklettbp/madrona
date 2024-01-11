namespace madrona {

math::Vector3 Navmesh::samplePoint(RandKey rnd)
{
    using namespace math;

    RandKey tbl_row_rnd = rand::split_i(rnd, 0);
    RandKey alias_p_rnd = rand::split_i(rnd, 1);
    RandKey bary_rnd = rand::split_i(rnd, 2);

    uint32_t tbl_row_idx = rand::sampleI32(tbl_row_rnd, 0, numTris);
    float p = rand::sampleUniform(alias_p_rnd);

    AliasEntry tbl_row = triSampleAliasTable[tbl_row_idx];

    uint32_t tri_idx = p < tbl_row.tau ? tbl_row_idx : tbl_row.alias;

    Vector3 a = vertices[triIndices[3 * tri_idx]];
    Vector3 b = vertices[triIndices[3 * tri_idx + 1]];
    Vector3 c = vertices[triIndices[3 * tri_idx + 2]];

    Vector2 uv = rand::sample2xUniform(bary_rnd);

    if (uv.x + uv.y > 1.f) {
        uv.x = 1.f - uv.x;
        uv.y = 1.f - uv.y;
    }

    float w = 1.f - uv.x - uv.y;

    return a * uv.x + b * uv.y + c * w;
}

}
