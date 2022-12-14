#include <madrona/importer.hpp>
#include <madrona/dyn_array.hpp>
#include <madrona/heap_array.hpp>

#include <string>
#include <fstream>
#include <charconv>

#include <meshoptimizer.h>

namespace madrona {
namespace imp {

using namespace math;

static bool loadOBJ(const char *path, ImportedObject &imported)
{
    using namespace std;

    ifstream file(path);

    struct ObjIDX {
        uint32_t posIdx;
        uint32_t normalIdx;
        uint32_t uvIdx;
    };

    constexpr CountT reserve_elems = 128;

    auto readVec2 = [](string_view str) {
        const char *start = str.begin();
        const char *end = str.end();

        while (*start == ' ' && start < end) {
            start += 1;
        }

        float x;
        auto res = from_chars(start, end, x);

        if (res.ptr == start) {
            FATAL("Failed to read x");
        }

        start = res.ptr;

        while (*start == ' ' && start < end) {
            start += 1;
        }

        float y;
        res = from_chars(start, end, y);

        if (res.ptr == start) {
            FATAL("Failed to read y");
        }

        return math::Vector2 { x, y };
    };

    auto readVec3 = [](string_view str) {
        const char *start = str.begin();
        const char *end = str.end();

        while (*start == ' ' && start < end) {
            start += 1;
        }

        float x;
        auto res = from_chars(start, end, x);

        if (res.ptr == start) {
            FATAL("Failed to read x");
        }

        start = res.ptr;

        while (*start == ' ' && start < end) {
            start += 1;
        }

        float y;
        res = from_chars(start, end, y);

        if (res.ptr == start) {
            FATAL("Failed to read y");
        }

        start = res.ptr;

        while (*start == ' ' && start < end) {
            start += 1;
        }

        float z;
        res = from_chars(start, end, z);

        if (res.ptr == start) {
            FATAL("Failed to read z");
        }

        return math::Vector3 { x, y, z };
    };

    auto readIdxTriple = [](const char *start, const char *end,
                            ObjIDX *idx_triple) {
        uint32_t pos_idx;
        auto res = from_chars(start, end, pos_idx);

        if (res.ptr == start) {
            FATAL("Failed to read position idx: %s", start);
        }

        start = res.ptr;

        if (start == end || start[0] != '/') {
            *idx_triple = {
                .posIdx = pos_idx,
                .normalIdx = 0,
                .uvIdx = 0,
            };

            return start;
        }

        start += 1;

        uint32_t uv_idx;

        if (start[0] == '/') {
            uv_idx = 0;
        } else {
            res = from_chars(start, end, uv_idx);

            if (res.ptr == start) {
                FATAL("Failed to read UV idx");
            }

            start = res.ptr;
        }

        if (start == end || start[0] != '/') {
            *idx_triple = {
                .posIdx = pos_idx,
                .normalIdx = 0,
                .uvIdx = uv_idx,
            };

            return start;
        }

        start += 1;

        uint32_t normal_idx;
        res = from_chars(start, end, normal_idx);

        if (res.ptr == start) {
            FATAL("Failed to read normal idx");
        }

        *idx_triple = {
            .posIdx = pos_idx,
            .normalIdx = normal_idx,
            .uvIdx = uv_idx,
        };

        return res.ptr;
    };

    DynArray<math::Vector3> positions(reserve_elems);
    DynArray<math::Vector3> normals(reserve_elems);
    DynArray<math::Vector2> uvs(reserve_elems);
    DynArray<ObjIDX> indices(reserve_elems);
    DynArray<uint32_t> face_counts(reserve_elems);
    
    DynArray<math::Vector3> unindexed_positions(reserve_elems);
    DynArray<math::Vector3> unindexed_normals(reserve_elems);
    DynArray<math::Vector2> unindexed_uvs(reserve_elems);
    DynArray<uint32_t> fake_indices(reserve_elems);

    auto commitMesh = [&]() {
        if (indices.size() == 0) {
            if (positions.size() > 0 || normals.size() > 0 || uvs.size() > 0) {
                return false;
            }

            return true;
        }

        // Unindex mesh
        for (const ObjIDX &obj_idx : indices) {
            fake_indices.push_back(uint32_t(unindexed_positions.size()));

            if (obj_idx.posIdx == 0) {
                return false;
            }

            int64_t pos_idx = obj_idx.posIdx - 1;
            if (pos_idx >= positions.size()) {
                return false;
            }

            unindexed_positions.push_back(positions[pos_idx]);

            if (obj_idx.normalIdx > 0) {
                int64_t normal_idx = obj_idx.normalIdx - 1;
                if (normal_idx >= normals.size() ) {
                    return false;
                }

                unindexed_normals.push_back(normals[normal_idx]);
            } else if (unindexed_normals.size() > 0) {
                return false;
            }

            if (obj_idx.uvIdx > 0) {
                int64_t uv_idx = obj_idx.uvIdx - 1;
                if (uv_idx >= uvs.size()) {
                    return false;
                }

                unindexed_uvs.push_back(uvs[uv_idx]);
            } else if (unindexed_uvs.size() > 0) {
                return false;
            }
        }

        std::array<meshopt_Stream, 3> vertex_streams;
        vertex_streams[0] = {
            .data = unindexed_positions.data(),
            .size = sizeof(Vector3),
            .stride = sizeof(Vector3),
        };

        int64_t num_vert_streams = 1;

        if (unindexed_normals.size() > 0) {
            vertex_streams[num_vert_streams++] = {
                .data = unindexed_normals.data(),
                .size = sizeof(Vector3),
                .stride = sizeof(Vector3),
            };
        }

        if (unindexed_uvs.size() > 0) {
            vertex_streams[num_vert_streams++] = {
                .data = unindexed_uvs.data(),
                .size = sizeof(Vector2),
                .stride = sizeof(Vector2),
            };
        }

        HeapArray<uint32_t> vertex_remap(unindexed_positions.size());

        CountT num_new_verts = meshopt_generateVertexRemapMulti(
            vertex_remap.data(), nullptr, vertex_remap.size(), 
            vertex_remap.size(), vertex_streams.data(), num_vert_streams);

        DynArray<Vector3> new_positions(0);
        new_positions.resize(num_new_verts, [](Vector3 *) {});
        DynArray<uint32_t> new_indices(0);
        new_indices.resize(fake_indices.size(), [](uint32_t *) {});
        DynArray<Vector3> new_normals(0);
        DynArray<Vector2> new_uvs(0);

        meshopt_remapVertexBuffer(new_positions.data(),
                                  unindexed_positions.data(),
                                  unindexed_positions.size(),
                                  sizeof(Vector3),
                                  vertex_remap.data());

        meshopt_remapIndexBuffer(new_indices.data(), fake_indices.data(),
                                 fake_indices.size(), vertex_remap.data());

        if (unindexed_normals.size() > 0) {
            new_normals.resize(num_new_verts, [](Vector3 *) {});
            meshopt_remapVertexBuffer(new_normals.data(),
                                      unindexed_normals.data(),
                                      unindexed_normals.size(),
                                      sizeof(Vector3),
                                      vertex_remap.data());
        }

        if (unindexed_uvs.size() > 0) {
            new_uvs.resize(num_new_verts, [](Vector2 *) {});

            meshopt_remapVertexBuffer(new_uvs.data(),
                                      unindexed_uvs.data(),
                                      unindexed_uvs.size(),
                                      sizeof(Vector2),
                                      vertex_remap.data());
        }

        DynArray<uint32_t> face_counts_copy(face_counts.size());

        bool fully_triangular = true;
        for (uint32_t c : face_counts) {
            if (c != 3){
                fully_triangular = false;
            }

            face_counts_copy.push_back(c);
        }

        positions.clear();
        normals.clear();
        uvs.clear();
        indices.clear();
        unindexed_positions.clear();
        unindexed_normals.clear();
        unindexed_uvs.clear();
        fake_indices.clear();
        face_counts.clear();

        imported.meshes.push_back({
            new_positions.data(),
            new_normals.data(),
            nullptr,
            new_uvs.data(),
            new_indices.data(),
            fully_triangular ? nullptr : face_counts_copy.data(),
            uint32_t(new_positions.size()),
            uint32_t(face_counts_copy.size()),
        });

        imported.positionArrays.emplace_back(std::move(new_positions));
        imported.normalArrays.emplace_back(std::move(new_normals));
        imported.uvArrays.emplace_back(std::move(new_uvs));
        imported.indexArrays.emplace_back(std::move(new_indices));

        if (!fully_triangular) {
            imported.faceCountArrays.emplace_back(std::move(face_counts_copy));
        }

        return true;
    };

    string line;
    while (getline(file, line)) {
        if (line[0] == '#') continue;

        if (line[0] == 'o') {
            bool valid = commitMesh();
            if (!valid) {
                return false;
            }
        }

        if (line[0] == 's') continue;

        if (line[0] == 'v') {
            if (line[1] == ' ') {
                math::Vector3 pos = readVec3(string_view(line).substr(1));
                positions.push_back(pos);
            } else if (line[1] == 'n') {
                math::Vector3 normal = readVec3(string_view(line).substr(2));
                normals.push_back(normal);
            } else if (line[1] == 't') {
                math::Vector2 uv = readVec2(string_view(line).substr(2));
                uvs.push_back(uv);
            }
        }

        if (line[0] == 'f') {
            const char *start = line.data() + 1;
            const char *end = line.data() + line.size();

            int64_t face_count = 0;
            while (true) {
                while (start < end && (*start == ' ' || *start == '\r')) {
                    start += 1;
                }

                if (start == end) {
                    break;
                }

                ObjIDX idx;
                start = readIdxTriple(start, end, &idx);
                indices.push_back(idx);

                face_count++;
            }

            if (face_count == 0) return false;
            face_counts.push_back(face_count);
        }
    }

    return commitMesh();
}

Optional<ImportedObject> ImportedObject::importObject(const char *path)
{
    ImportedObject imported {
        .positionArrays = DynArray<DynArray<Vector3>>(1),
        .normalArrays = DynArray<DynArray<Vector3>>(1),
        .tangentAndSignArrays = DynArray<DynArray<Vector4>>(1),
        .uvArrays = DynArray<DynArray<Vector2>>(1),
        .indexArrays = DynArray<DynArray<uint32_t>>(1),
        .faceCountArrays = DynArray<DynArray<uint32_t>>(1),
        .meshes = DynArray<SourceMesh>(1),
    };

    std::string_view path_view(path);

    auto extension_pos = path_view.rfind('.');
    if (extension_pos == path_view.npos) {
        return Optional<ImportedObject>::none();
    }
    auto extension = path_view.substr(extension_pos + 1);

    if (extension == "obj") {
        loadOBJ(path, imported);
    }

    return imported;
}

}
}
