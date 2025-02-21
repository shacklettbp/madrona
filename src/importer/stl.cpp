#include "stl.hpp"

#include <fstream>
#include <meshoptimizer.h>

namespace madrona::imp {

struct STLLoader::Impl {
    
};
    
STLLoader::STLLoader(Span<char> err_buf)
{
}

STLLoader::~STLLoader()
{
}

bool STLLoader::load(const char *path, ImportedAssets &imported_assets)
{
    using namespace math;

    std::string file_path(path);

    std::ifstream file(file_path, std::ios::binary | std::ios::ate);

    if (!file.is_open() || !file.good()) {
        assert(false);
        return false;
    }

    uint32_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(file_size);
    if (!file.read((char *)buffer.data(), file_size)) {
        assert(false);
        return false;
    }

    uint8_t *ptr = buffer.data() + 80;
    uint32_t num_triangles = *((uint32_t *)ptr);

    ptr += sizeof(uint32_t);

    std::vector<Vector3> normals;
    std::vector<Vector3> unindexed_vertices;
    
    normals.resize(num_triangles);
    unindexed_vertices.resize(num_triangles * 3);

    for (uint32_t tri_idx = 0; tri_idx < num_triangles; ++tri_idx) {
        Vector3 *data = (Vector3 *)ptr;

        normals[tri_idx] = data[0];
        unindexed_vertices[tri_idx * 3 + 0] = data[1];
        unindexed_vertices[tri_idx * 3 + 1] = data[2];
        unindexed_vertices[tri_idx * 3 + 2] = data[3];

        ptr += sizeof(Vector3) * 4 + sizeof(uint16_t);
    }

    uint32_t index_count = num_triangles * 3;
    uint32_t unindexed_vertex_count = num_triangles * 3;

    std::vector<uint32_t> remap;
    remap.resize(unindexed_vertex_count);

    uint32_t vertex_count = meshopt_generateVertexRemap(
            remap.data(),
            nullptr,
            index_count,
            unindexed_vertices.data(),
            unindexed_vertex_count,
            sizeof(Vector3));

    DynArray<uint32_t> indices(index_count);
    indices.resize(index_count, [](uint32_t *) {});

    DynArray<Vector3> vertices(vertex_count);
    vertices.resize(vertex_count, [](Vector3 *) {});

    meshopt_remapIndexBuffer(
            indices.data(),
            nullptr,
            index_count,
            remap.data());

    meshopt_remapVertexBuffer(
            vertices.data(),
            unindexed_vertices.data(),
            unindexed_vertex_count,
            sizeof(Vector3),
            remap.data());

    DynArray<SourceMesh> mesh(1);
    mesh.resize(1, [](SourceMesh *) {});

    mesh[0] = SourceMesh {
        .positions = vertices.data(),
        .normals = nullptr,
        .tangentAndSigns = nullptr,
        .uvs = nullptr,
        .indices = indices.data(),
        .faceCounts = nullptr,
        .faceMaterials = nullptr,
        .numVertices = vertex_count,
        .numFaces = num_triangles,
        .materialIDX = 0xFFFF'FFFF,
        .name = "",
    };

    imported_assets.geoData.positionArrays.push_back(
            std::move(vertices));
    imported_assets.geoData.indexArrays.push_back(
            std::move(indices));
    imported_assets.objects.push_back(
            SourceObject {
                .meshes = { mesh.data(), 1 }
            });
    imported_assets.geoData.meshArrays.push_back(
            std::move(mesh));

    return true;
}

}
