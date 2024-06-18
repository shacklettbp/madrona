#include "obj.hpp"

#include <cstdarg>
#include <charconv>
#include <fast_float/fast_float.h>
#include <fstream>
#include <string>
#include <inttypes.h>

#include <meshoptimizer.h>

#include <madrona/heap_array.hpp>

namespace madrona::imp {

using namespace madrona::math;

namespace {

struct ObjIDX {
    uint32_t posIdx;
    uint32_t normalIdx;
    uint32_t uvIdx;
};

}

struct OBJLoader::Impl {
    DynArray<SourceMesh> objMeshes;

    // These buffers are iteratively filled while parsing the OBJ
    DynArray<math::Vector3> curPositions;
    DynArray<math::Vector3> curNormals;
    DynArray<math::Vector2> curUVs;
    DynArray<ObjIDX> curIndices;
    DynArray<uint32_t> curFaceCounts;

    // Temporary buffers kept around to avoid reallocations
    DynArray<math::Vector3> unindexedPositions;
    DynArray<math::Vector3> unindexedNormals;
    DynArray<math::Vector2> unindexedUVs;
    DynArray<uint32_t> fakeIndices;
    DynArray<uint32_t> vertexRemap;

    // Extra data for error reporting
    Span<char> errBuf;
    const char *filePath;
    const char *curSrcLine;
    int64_t curSrcLineIdx;

    Impl(Span<char> err_buf);

    void setLine(const char *src_line, int64_t line_idx);

    void recordError(const char *fmt_string, ...) const;

    bool commitMesh(ImportedAssets &out_assets);

    bool load(const char *path, ImportedAssets &imported_assets);

    static constexpr inline CountT reserve_elems = 128;
};

using LoaderData = OBJLoader::Impl;

namespace {

inline fast_float::from_chars_result fromCharsFloat(
    const char *first,
    const char *last,
    float &value,
    fast_float::chars_format fmt = fast_float::chars_format::general)
{
    return fast_float::from_chars(first, last, value, fmt);
}

inline std::from_chars_result fromCharsU32(
    const char *first,
    const char *last,
    uint32_t &value,
    int base = 10)
{
    return std::from_chars(first, last, value, base);
}

inline bool parseVec2(std::string_view str,
                      math::Vector2 *out,
                      const LoaderData &loader)
{
    const char *start = str.data();
    const char *end = start + str.size();

    while (*start == ' ' && start < end) {
        start += 1;
    }

    float x;
    auto res = fromCharsFloat(start, end, x);

    if (res.ptr == start) {
        loader.recordError("Failed to read x component.");
        return false;
    }

    start = res.ptr;

    while (*start == ' ' && start < end) {
        start += 1;
    }

    float y;
    res = fromCharsFloat(start, end, y);

    if (res.ptr == start) {
        loader.recordError("Failed to read y component.");
        return false;
    }

    *out = math::Vector2 { x, y };
    return true;
};


inline bool parseVec3(std::string_view str,
                      math::Vector3 *out,
                      const LoaderData &loader)
{
    const char *start = str.data();
    const char *end = start + str.size();

    while (*start == ' ' && start < end) {
        start += 1;
    }

    float x;
    auto res = fromCharsFloat(start, end, x);

    if (res.ptr == start) {
        loader.recordError("Failed to read x component.");
        return false;
    }

    start = res.ptr;

    while (*start == ' ' && start < end) {
        start += 1;
    }

    float y;
    res = fromCharsFloat(start, end, y);

    if (res.ptr == start) {
        loader.recordError("Failed to read y component.");
        return false;
    }

    start = res.ptr;

    while (*start == ' ' && start < end) {
        start += 1;
    }

    float z;
    res = fromCharsFloat(start, end, z);

    if (res.ptr == start) {
        loader.recordError("Failed to read z component.");
        return false;
    }

    *out = math::Vector3 { x, y, z };
    return true;
};

inline bool parseIdxTriple(const char *start, const char *end,
                           ObjIDX *idx_triple, const char **next,
                           const LoaderData &loader)
{
    uint32_t pos_idx;
    auto res = fromCharsU32(start, end, pos_idx);

    if (res.ptr == start) {
        loader.recordError("Failed to read position idx: %s.", start);
        return false;
    }

    start = res.ptr;

    if (start == end || start[0] != '/') {
        *idx_triple = {
            .posIdx = pos_idx,
            .normalIdx = 0,
            .uvIdx = 0,
        };

        *next = start;
        return true;
    }

    start += 1;

    uint32_t uv_idx;

    if (start[0] == '/') {
        uv_idx = 0;
    } else {
        res = fromCharsU32(start, end, uv_idx);

        if (res.ptr == start) {
            loader.recordError("Failed to read UV idx.");
            return false;
        }

        start = res.ptr;
    }

    if (start == end || start[0] != '/') {
        *idx_triple = {
            .posIdx = pos_idx,
            .normalIdx = 0,
            .uvIdx = uv_idx,
        };

        *next = start;
        return true;
    }

    start += 1;

    uint32_t normal_idx;
    res = fromCharsU32(start, end, normal_idx);

    if (res.ptr == start) {
        loader.recordError("Failed to read normal idx");
        return false;
    }

    *idx_triple = {
        .posIdx = pos_idx,
        .normalIdx = normal_idx,
        .uvIdx = uv_idx,
    };

    *next = res.ptr;

    return true;
};

}

OBJLoader::Impl::Impl(Span<char> err_buf)
    : objMeshes(1),
      curPositions(reserve_elems),
      curNormals(reserve_elems),
      curUVs(reserve_elems),
      curIndices(reserve_elems),
      curFaceCounts(reserve_elems),
      unindexedPositions(reserve_elems),
      unindexedNormals(reserve_elems),
      unindexedUVs(reserve_elems),
      fakeIndices(reserve_elems),
      vertexRemap(reserve_elems),
      errBuf(err_buf),
      filePath(nullptr),
      curSrcLine(nullptr),
      curSrcLineIdx(-1)
{}

void OBJLoader::Impl::setLine(const char *src_line, int64_t line_idx)
{
    curSrcLine = src_line;
    curSrcLineIdx = line_idx;
}

void OBJLoader::Impl::recordError(const char *fmt_string, ...) const
{
    if (errBuf.data() == nullptr) {
        return;
    }

    int prefix_chars_written;

    if (curSrcLine == nullptr) {
        prefix_chars_written = snprintf(errBuf.data(), errBuf.size(),
            "Invalid OBJ File %s: ", filePath);
    } else {
        prefix_chars_written = snprintf(errBuf.data(), errBuf.size(),
            "Invalid OBJ File %s, line %" PRIi64 "\n%s\n", filePath,
            curSrcLineIdx, curSrcLine);
    }

    if (prefix_chars_written < errBuf.size()) {
        va_list args;
        va_start(args, fmt_string);

        size_t remaining_data = errBuf.size() - prefix_chars_written;
        vsnprintf(errBuf.data() + prefix_chars_written, remaining_data,
                  fmt_string, args);
    }
}

bool OBJLoader::Impl::commitMesh(ImportedAssets &out_assets)
{
    if (curIndices.size() == 0) {
        if (curPositions.size() > 0 || curNormals.size() > 0 ||
                curUVs.size() > 0) {
            recordError("Unindexed meshes not supported");
            return false;
        }

        // Just do nothing for a totally empty mesh
        return true;
    }

    // Unindex mesh
    for (const ObjIDX &obj_idx : curIndices) {
        fakeIndices.push_back(uint32_t(unindexedPositions.size()));

        if (obj_idx.posIdx == 0) {
            recordError("Missing position index");
            return false;
        }

        int64_t pos_idx = obj_idx.posIdx - 1;
        if (pos_idx >= curPositions.size()) {
            recordError("Out of range position index %" PRIi64 ".", pos_idx);
            return false;
        }

        unindexedPositions.push_back(curPositions[pos_idx]);

        if (obj_idx.normalIdx > 0) {
            int64_t normal_idx = obj_idx.normalIdx - 1;
            if (normal_idx >= curNormals.size() ) {
                recordError("Out of range normal index %" PRIi64 ".",
                            normal_idx);
                return false;
            }

            unindexedNormals.push_back(curNormals[normal_idx]);
        } else if (curNormals.size() > 0) {
            recordError("Missing normal index.");
            return false;
        }

        if (obj_idx.uvIdx > 0) {
            int64_t uv_idx = obj_idx.uvIdx - 1;
            if (uv_idx >= curUVs.size()) {
                recordError("Out of range UV index %" PRIi64 ".",
                            uv_idx);
                return false;
            }

            unindexedUVs.push_back(curUVs[uv_idx]);
        } else if (curUVs.size() > 0) {
            recordError("Missing UV index.");
            return false;
        }
    }

    std::array<meshopt_Stream, 3> vertex_streams;
    vertex_streams[0] = {
        .data = unindexedPositions.data(),
        .size = sizeof(Vector3),
        .stride = sizeof(Vector3),
    };

    int64_t num_vert_streams = 1;

    if (unindexedNormals.size() > 0) {
        vertex_streams[num_vert_streams++] = {
            .data = unindexedNormals.data(),
            .size = sizeof(Vector3),
            .stride = sizeof(Vector3),
        };
    }

    if (unindexedUVs.size() > 0) {
        vertex_streams[num_vert_streams++] = {
            .data = unindexedUVs.data(),
            .size = sizeof(Vector2),
            .stride = sizeof(Vector2),
        };
    }

    // OBJ files have separate indices for each attribute. Use
    // meshoptimizer to reindex the mesh into a single vertex stream.
    vertexRemap.resize(unindexedPositions.size(), [](uint32_t *) {});

    CountT num_new_verts = meshopt_generateVertexRemapMulti(
        vertexRemap.data(), nullptr, vertexRemap.size(), 
        vertexRemap.size(), vertex_streams.data(), num_vert_streams);

    DynArray<Vector3> new_positions(0);
    new_positions.resize(num_new_verts, [](Vector3 *) {});
    DynArray<uint32_t> new_indices(0);
    new_indices.resize(fakeIndices.size(), [](uint32_t *) {});
    DynArray<Vector3> new_normals(0);
    DynArray<Vector2> new_uvs(0);

    meshopt_remapVertexBuffer(new_positions.data(),
                              unindexedPositions.data(),
                              unindexedPositions.size(),
                              sizeof(Vector3),
                              vertexRemap.data());

    meshopt_remapIndexBuffer(new_indices.data(), fakeIndices.data(),
                             fakeIndices.size(), vertexRemap.data());

    if (unindexedNormals.size() > 0) {
        new_normals.resize(num_new_verts, [](Vector3 *) {});
        meshopt_remapVertexBuffer(new_normals.data(),
                                  unindexedNormals.data(),
                                  unindexedNormals.size(),
                                  sizeof(Vector3),
                                  vertexRemap.data());
    }

    if (unindexedUVs.size() > 0) {
        new_uvs.resize(num_new_verts, [](Vector2 *) {});

        meshopt_remapVertexBuffer(new_uvs.data(),
                                  unindexedUVs.data(),
                                  unindexedUVs.size(),
                                  sizeof(Vector2),
                                  vertexRemap.data());
    }

    DynArray<uint32_t> face_counts_copy(curFaceCounts.size());

    bool fully_triangular = true;
    for (uint32_t c : curFaceCounts) {
        if (c != 3){
            fully_triangular = false;
        }

        face_counts_copy.push_back(c);
    }

    curIndices.clear();
    curFaceCounts.clear();
    unindexedPositions.clear();
    unindexedNormals.clear();
    unindexedUVs.clear();
    fakeIndices.clear();
    vertexRemap.clear();

    objMeshes.push_back({
        .positions = new_positions.data(),
        .normals = new_normals.data(),
        .tangentAndSigns = nullptr,
        .uvs = new_uvs.data(),
        .indices = new_indices.data(),
        .faceCounts = fully_triangular ? nullptr : face_counts_copy.data(),
        .faceMaterials = nullptr,
        .numVertices = uint32_t(new_positions.size()),
        .numFaces = uint32_t(face_counts_copy.size()),
        .materialIDX = 0xFFFF'FFFF,
    });

    out_assets.geoData.positionArrays.emplace_back(
        std::move(new_positions));
    out_assets.geoData.normalArrays.emplace_back(
        std::move(new_normals));
    out_assets.geoData.uvArrays.emplace_back(
        std::move(new_uvs));
    out_assets.geoData.indexArrays.emplace_back(
        std::move(new_indices));

    if (!fully_triangular) {
        out_assets.geoData.faceCountArrays.emplace_back(
            std::move(face_counts_copy));
    }

    return true;
}

bool OBJLoader::Impl::load(const char *path, ImportedAssets &imported_assets)
{
    using std::string_view;

    // These arrays aren't cleared incrementally because OBJs are indexed
    // from the start of the file. Only clear them here, at the start to make
    // sure all indices correctly start at 1.
    curPositions.clear();
    curNormals.clear();
    curUVs.clear();

    filePath = path;

    std::ifstream file(path);
    if (!file.is_open() || !file.good()) {
        recordError("Could not open.");
        return false;
    }

    std::string line;
    int64_t line_idx = 1;
    while (std::getline(file, line)) {
        setLine(line.c_str(), line_idx++);

        if (line[0] == '#') continue;

        if (line[0] == 'o') {
            bool valid = commitMesh(imported_assets);
            if (!valid) {
                return false;
            }
        }

        if (line[0] == 's') continue;

        if (line[0] == 'v') {
            if (line[1] == ' ') {
                math::Vector3 pos;
                bool valid = parseVec3(string_view(line).substr(1), &pos,
                                       *this);
                if (!valid) return false;

                curPositions.push_back(pos);
            } else if (line[1] == 'n') {
                math::Vector3 normal;
                bool valid = parseVec3(string_view(line).substr(2), &normal,
                                       *this);
                if (!valid) return false;

                curNormals.push_back(normal);
            } else if (line[1] == 't') {
                math::Vector2 uv;
                bool valid = parseVec2(string_view(line).substr(2), &uv,
                                       *this);
                if (!valid) return false;

                curUVs.push_back(uv);
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
                const char *next;
                bool valid = parseIdxTriple(start, end, &idx, &next,
                                            *this);
                if (!valid) return false;

                start = next;

                curIndices.push_back(idx);

                face_count++;
            }

            if (face_count == 0) {
                recordError("Face with no indices.");
                return false;
            } 

            curFaceCounts.push_back(face_count);
        }
    }

    if (!commitMesh(imported_assets)) {
        return false;
    }

    imported_assets.objects.push_back({
        .meshes = { objMeshes.data(), objMeshes.size() },
    });

    imported_assets.geoData.meshArrays.emplace_back(
        std::move(objMeshes));

    return true;
}

OBJLoader::OBJLoader(Span<char> err_buf)
    : impl_(new Impl(err_buf))
{}

OBJLoader::~OBJLoader() {}

bool OBJLoader::load(const char *path, ImportedAssets &imported_assets)
{
    return impl_->load(path, imported_assets);
}

}
