#pragma once

#include <madrona/heap_array.hpp>
#include <madrona/span.hpp>
#include <madrona/importer.hpp>

# define PI 3.14159265358979323846

using namespace madrona;
using namespace madrona::imp;

namespace madMJX
{

SourceMesh CreatePlane(ImportedAssets &assets, float size = 1.0)
{
    DynArray<math::Vector3> positions = {
        math::Vector3(size, -size, 0.0),
        math::Vector3(-size, size, 0.0),
        math::Vector3(-size, -size, 0.0),
        math::Vector3(size, size, 0.0)};

    DynArray<math::Vector3> normals = {
        math::Vector3(0.0, 0.0, 1.0),
        math::Vector3(0.0, 0.0, 1.0),
        math::Vector3(0.0, 0.0, 1.0),
        math::Vector3(0.0, 0.0, 1.0)};

    DynArray<math::Vector2> uvs = {
        math::Vector2(1.0, 0.0),
        math::Vector2(0.0, 1.0),
        math::Vector2(0.0, 0.0),
        math::Vector2(1.0, 1.0)};

    DynArray<uint32_t> indices = {
        0, 1, 2,
        0, 3, 1};

    SourceMesh mesh = {
        .positions = positions.data(),
        .normals = normals.data(),
        .tangentAndSigns = nullptr,
        .uvs = uvs.data(),
        .indices = indices.data(),
        .faceCounts = nullptr,
        .faceMaterials = nullptr,
        .numVertices = 4,
        .numFaces = 2,
        .materialIDX = 0,
    };

    assets.geoData.positionArrays.emplace_back(std::move(positions));
    assets.geoData.normalArrays.emplace_back(std::move(normals));
    assets.geoData.uvArrays.emplace_back(std::move(uvs));
    assets.geoData.indexArrays.emplace_back(std::move(indices));
    return mesh;
}

SourceMesh CreateBox(ImportedAssets &assets, float size = 1.0)
{
    DynArray<math::Vector3> positions = {
        math::Vector3(-size, size, size),
        math::Vector3(size, -size, size),
        math::Vector3(size, size, size),
        math::Vector3(size, -size, size),
        math::Vector3(-size, -size, -size),
        math::Vector3(size, -size, -size),
        math::Vector3(-size, -size, size),
        math::Vector3(-size, size, -size),
        math::Vector3(-size, -size, -size),
        math::Vector3(size, size, -size),
        math::Vector3(-size, -size, -size),
        math::Vector3(-size, size, -size),
        math::Vector3(size, size, size),
        math::Vector3(size, -size, -size),
        math::Vector3(size, size, -size),
        math::Vector3(-size, size, size),
        math::Vector3(size, size, -size),
        math::Vector3(-size, size, -size),
        math::Vector3(-size, -size, size),
        math::Vector3(-size, -size, size),
        math::Vector3(-size, size, size),
        math::Vector3(size, -size, -size),
        math::Vector3(size, -size, size),
        math::Vector3(size, size, size)};

    DynArray<math::Vector3> normals = {
        math::Vector3(0.0, 0.0, 1.0),
        math::Vector3(0.0, 0.0, 1.0),
        math::Vector3(0.0, 0.0, 1.0),
        math::Vector3(0.0, -1.0, 0.0),
        math::Vector3(0.0, -1.0, 0.0),
        math::Vector3(0.0, -1.0, 0.0),
        math::Vector3(-1.0, 0.0, 0.0),
        math::Vector3(-1.0, 0.0, 0.0),
        math::Vector3(-1.0, 0.0, 0.0),
        math::Vector3(0.0, 0.0, -1.0),
        math::Vector3(0.0, 0.0, -1.0),
        math::Vector3(0.0, 0.0, -1.0),
        math::Vector3(1.0, 0.0, 0.0),
        math::Vector3(1.0, 0.0, 0.0),
        math::Vector3(1.0, 0.0, 0.0),
        math::Vector3(0.0, 1.0, 0.0),
        math::Vector3(0.0, 1.0, 0.0),
        math::Vector3(0.0, 1.0, 0.0),
        math::Vector3(0.0, 0.0, 1.0),
        math::Vector3(0.0, -1.0, 0.0),
        math::Vector3(-1.0, 0.0, 0.0),
        math::Vector3(0.0, 0.0, -1.0),
        math::Vector3(1.0, 0.0, 0.0),
        math::Vector3(0.0, 1.0, 0.0)};

    DynArray<math::Vector2> uvs = {
        math::Vector2(0.875, 0.5),
        math::Vector2(0.625, 0.75),
        math::Vector2(0.625, 0.5),
        math::Vector2(0.625, 0.75),
        math::Vector2(0.375, 1.0),
        math::Vector2(0.375, 0.75),
        math::Vector2(0.625, 0.0),
        math::Vector2(0.375, 0.25),
        math::Vector2(0.375, 0.0),
        math::Vector2(0.375, 0.5),
        math::Vector2(0.125, 0.75),
        math::Vector2(0.125, 0.5),
        math::Vector2(0.625, 0.5),
        math::Vector2(0.375, 0.75),
        math::Vector2(0.375, 0.5),
        math::Vector2(0.625, 0.25),
        math::Vector2(0.375, 0.5),
        math::Vector2(0.375, 0.25),
        math::Vector2(0.875, 0.75),
        math::Vector2(0.625, 1.0),
        math::Vector2(0.625, 0.25),
        math::Vector2(0.375, 0.75),
        math::Vector2(0.625, 0.75),
        math::Vector2(0.625, 0.5)};

    DynArray<uint32_t> indices = {
        0, 1, 2,
        3, 4, 5,
        6, 7, 8,
        9, 10, 11,
        12, 13, 14,
        15, 16, 17,
        0, 18, 1,
        3, 19, 4,
        6, 20, 7,
        9, 21, 10,
        12, 22, 13,
        15, 23, 16};

    SourceMesh mesh = {
        .positions = positions.data(),
        .normals = normals.data(),
        .tangentAndSigns = nullptr,
        .uvs = uvs.data(),
        .indices = indices.data(),
        .faceCounts = nullptr,
        .faceMaterials = nullptr,
        .numVertices = 24,
        .numFaces = 12,
        .materialIDX = 0,
    };

    assets.geoData.positionArrays.emplace_back(std::move(positions));
    assets.geoData.normalArrays.emplace_back(std::move(normals));
    assets.geoData.uvArrays.emplace_back(std::move(uvs));
    assets.geoData.indexArrays.emplace_back(std::move(indices));
    return mesh;
}

SourceMesh CreateSphere(
    ImportedAssets &assets,
    float radius = 1.0,
    int subdivisions = 16)
{
    DynArray<math::Vector3> positions(0);
    DynArray<math::Vector3> normals(0);
    DynArray<math::Vector2> uvs(0);
    DynArray<uint32_t> indices(0);
    positions.reserve((subdivisions + 1) * (subdivisions + 1));
    normals.reserve((subdivisions + 1) * (subdivisions + 1));
    uvs.reserve((subdivisions + 1) * (subdivisions + 1));
    indices.reserve(subdivisions * subdivisions * 6);


    // Create a sphere programmatically. Use triangles to create the faces.
    for (int i = 0; i <= subdivisions; i++)
    {
        float theta = (float)i * PI / (float)subdivisions;
        float sin_theta = sin(theta);
        float cos_theta = cos(theta);

        for (int j = 0; j <= subdivisions; j++)
        {
            float phi = (float)j * 2 * PI / (float)subdivisions;
            float sin_phi = sin(phi);
            float cos_phi = cos(phi);

            float x = radius * cos_phi * sin_theta;
            float y = radius * cos_theta;
            float z = radius * sin_phi * sin_theta;

            float nx = cos_phi * sin_theta;
            float ny = cos_theta;
            float nz = sin_phi * sin_theta;

            positions.push_back(math::Vector3(x, y, z));
            normals.push_back(math::Vector3(nx, ny, nz));
            uvs.push_back(math::Vector2((float)j/subdivisions, (float)i/subdivisions));
        }
    }

    for (int i = 0; i < subdivisions; i++)
    {
        for (int j = 0; j < subdivisions; j++)
        {
            int first = i * (subdivisions + 1) + j;
            int second = first + subdivisions + 1;
            indices.push_back(first);
            indices.push_back(first + 1);
            indices.push_back(second);

            indices.push_back(second);
            indices.push_back(first + 1);
            indices.push_back(second + 1);
        }
    }

    SourceMesh mesh = {
        .positions = positions.data(),
        .normals = normals.data(),
        .tangentAndSigns = nullptr,
        .uvs = uvs.data(),
        .indices = indices.data(),
        .faceCounts = nullptr,
        .faceMaterials = nullptr,
        .numVertices = (uint32_t)positions.size(),
        .numFaces = (uint32_t)indices.size() / 3,
        .materialIDX = 0,
    };

    assets.geoData.positionArrays.emplace_back(std::move(positions));
    assets.geoData.normalArrays.emplace_back(std::move(normals));
    assets.geoData.uvArrays.emplace_back(std::move(uvs));
    assets.geoData.indexArrays.emplace_back(std::move(indices));

    return mesh;
}

SourceMesh CreateCylinder(
    ImportedAssets &assets,
    float radius = 1.0,
    float height = 1.0,
    int subdivisions = 16)
{
    // Use the above code to create a cylinder
    DynArray<math::Vector3> positions(0);
    DynArray<math::Vector3> normals(0);
    DynArray<math::Vector2> uvs(0);
    DynArray<uint32_t> indices(0);

    int vertex_count = ((subdivisions + 1) * (subdivisions + 1)) + (2 * (subdivisions + 1));
    int index_count = subdivisions * subdivisions * 6 + (subdivisions * 6);

    positions.reserve(vertex_count);
    normals.reserve(vertex_count);
    uvs.reserve(vertex_count);
    indices.reserve(index_count);

    // Generate vertices, normals, and UVs for the side surface
    for (int y = 0; y <= subdivisions; ++y) {
        for (int x = 0; x <= subdivisions; ++x) {
            float x_seg = (float)x / subdivisions;
            float y_seg = (float)y / subdivisions;

            float vx = radius * cos(x_seg * 2.0f * PI);
            float vy = y_seg * height - height / 2.0f;
            float vz = radius * sin(x_seg * 2.0f * PI);

            positions.push_back(math::Vector3(vx, vz, vy));
            normals.push_back(math::Vector3(vx, 0.0f, vz));
            uvs.push_back(math::Vector2(x_seg, y_seg));
        }
    }

    // Generate indices for the side surface
    for (int y = 0; y < subdivisions; ++y) {
        for (int x = 0; x < subdivisions; ++x) {
            int current = y * (subdivisions + 1) + x;
            int next = current + 1;
            int next_row = current + subdivisions + 1;

            indices.push_back(current);
            indices.push_back(next);
            indices.push_back(next_row);

            indices.push_back(next);
            indices.push_back(next_row + 1);
            indices.push_back(next_row );
        }
    }

    // Generate vertices, normals, and UVs for the top cap
    for (int x = 0; x <= subdivisions; ++x) {
        float x_seg = (float)x / subdivisions;
        float vx = radius * cos(x_seg * 2.0f * PI);
        float vy = height / 2.0f;
        float vz = radius * sin(x_seg * 2.0f * PI);

        positions.push_back(math::Vector3(vx, vz, vy));
        normals.push_back(math::Vector3(0.0f, 1.0f, 0.0f));
        uvs.push_back(math::Vector2(x_seg, 1.0f));
    }

    // Generate indices for the top cap
    int top_cap_index_offset = positions.size() - subdivisions - 1;
    for (int x = 0; x < subdivisions; ++x) {
        indices.push_back(top_cap_index_offset);
        indices.push_back(top_cap_index_offset + x);
        indices.push_back(top_cap_index_offset + x + 1);
    }

    // Generate vertices, normals, and UVs for the bottom cap
    for (int x = 0; x <= subdivisions; ++x) {
        float x_seg = (float)x / subdivisions;
        float vx = radius * cos(x_seg * 2.0f * PI);
        float vy = -height / 2.0f;
        float vz = radius * sin(x_seg * 2.0f * PI);

        positions.push_back(math::Vector3(vx, vz, vy));
        normals.push_back(math::Vector3(0.0f, -1.0f, 0.0f));
        uvs.push_back(math::Vector2(x_seg, 0.0f));
    }

    // Generate indices for the bottom cap
    int bottom_cap_index_offset = positions.size() - subdivisions - 1;
    for (int x = 0; x < subdivisions; ++x) {
        indices.push_back(bottom_cap_index_offset + x);
        indices.push_back(bottom_cap_index_offset);
        indices.push_back(bottom_cap_index_offset + x + 1);
    }

    SourceMesh mesh = {
        .positions = positions.data(),
        .normals = normals.data(),
        .tangentAndSigns = nullptr,
        .uvs = uvs.data(),
        .indices = indices.data(),
        .faceCounts = nullptr,
        .faceMaterials = nullptr,
        .numVertices = (uint32_t)positions.size(),
        .numFaces = (uint32_t)indices.size() / 3,
        .materialIDX = 0,
    };

    assets.geoData.positionArrays.emplace_back(std::move(positions));
    assets.geoData.normalArrays.emplace_back(std::move(normals));
    assets.geoData.uvArrays.emplace_back(std::move(uvs));
    assets.geoData.indexArrays.emplace_back(std::move(indices));

    return mesh;
}

SourceMesh CreateCapsule(
    ImportedAssets &assets,
    float radius = 1.0,
    float height = 1.0,
    int subdivisions = 16)
{
    DynArray<math::Vector3> positions(0);
    DynArray<math::Vector3> normals(0);
    DynArray<math::Vector2> uvs(0);
    DynArray<uint32_t> indices(0);

    // Top Sphere
    for (int i = 0; i <= subdivisions; i++)
    {
        float theta = (float)i * PI / ((float)subdivisions * 2);
        float sin_theta = sin(theta);
        float cos_theta = cos(theta);

        for (int j = 0; j <= subdivisions; j++)
        {
            float phi = (float)j * 2 * PI / (float)subdivisions;
            float sin_phi = sin(phi);
            float cos_phi = cos(phi);

            float x = radius * cos_phi * sin_theta;
            float y = (height / 2) + (radius * cos_theta);
            float z = radius * sin_phi * sin_theta;

            float nx = cos_phi * sin_theta;
            float ny = cos_theta;
            float nz = sin_phi * sin_theta;

            positions.push_back(math::Vector3(x, z, y));
            normals.push_back(math::Vector3(nx, nz, ny));
            uvs.push_back(math::Vector2((float)j/subdivisions, (float)i/(2 * subdivisions)));
        }
    }

    for (int i = 0; i < subdivisions; i++)
    {
        for (int j = 0; j < subdivisions; j++)
        {
            int first = i * (subdivisions + 1) + j;
            int second = first + subdivisions + 1;
            indices.push_back(first);
            indices.push_back(second);
            indices.push_back(first + 1);

            indices.push_back(second);
            indices.push_back(second + 1);
            indices.push_back(first + 1);
        }
    }

    // Bottom Sphere
    for (int i = 0; i <= subdivisions; i++)
    {
        float theta = (float)i * PI / ((float)subdivisions * 2);
        float sin_theta = sin(theta);
        float cos_theta = cos(theta);

        for (int j = 0; j <= subdivisions; j++)
        {
            float phi = (float)j * 2 * PI / (float)subdivisions;
            float sin_phi = sin(phi);
            float cos_phi = cos(phi);

            float x = radius * cos_phi * sin_theta;
            float y = (-height / 2) - (radius * cos_theta);
            float z = radius * sin_phi * sin_theta;

            float nx = -1 * cos_phi * sin_theta;
            float ny = -1 * cos_theta;
            float nz = -1 * sin_phi * sin_theta;

            positions.push_back(math::Vector3(x, z, y));
            normals.push_back(math::Vector3(nx, nz, ny));
            uvs.push_back(math::Vector2((float)j/subdivisions, (float)i/(2 * subdivisions)));
        }
    }

    int bottomSphereIndex = positions.size() / 2;
    for (int i = 0; i < subdivisions; i++)
    {
        for (int j = 0; j < subdivisions; j++)
        {
            int first = i * (subdivisions + 1) + j + bottomSphereIndex;
            int second = first + subdivisions + 1;
            indices.push_back(first);
            indices.push_back(first + 1);
            indices.push_back(second);

            indices.push_back(second);
            indices.push_back(first + 1);
            indices.push_back(second + 1);
        }
    }

    // Generate vertices, normals, and UVs for the side surface
    for (int y = 0; y <= subdivisions; ++y) {
        for (int x = 0; x <= subdivisions; ++x) {
            float x_seg = (float)x / subdivisions;
            float y_seg = (float)y / subdivisions;

            float vx = radius * cos(x_seg * 2.0f * PI);
            float vy = y_seg * height - height / 2.0f;
            float vz = radius * sin(x_seg * 2.0f * PI);

            positions.push_back(math::Vector3(vx, vz, vy));
            normals.push_back(math::Vector3(vx, 0.0f, vz));
            uvs.push_back(math::Vector2(x_seg, y_seg));
        }
    }

    int postSphereIndex = bottomSphereIndex * 2;
    // Generate indices for the side surface
    for (int y = 0; y < subdivisions; ++y) {
        for (int x = 0; x < subdivisions; ++x) {
            int current = y * (subdivisions + 1) + x + postSphereIndex;
            int next = current + 1;
            int next_row = current + subdivisions + 1;

            indices.push_back(current);
            indices.push_back(next);
            indices.push_back(next_row);

            indices.push_back(next);
            indices.push_back(next_row + 1);
            indices.push_back(next_row );
        }
    }

    SourceMesh mesh = {
        .positions = positions.data(),
        .normals = normals.data(),
        .tangentAndSigns = nullptr,
        .uvs = uvs.data(),
        .indices = indices.data(),
        .faceCounts = nullptr,
        .faceMaterials = nullptr,
        .numVertices = (uint32_t)positions.size(),
        .numFaces = (uint32_t)indices.size() / 3,
        .materialIDX = 0,
    };

    assets.geoData.positionArrays.emplace_back(std::move(positions));
    assets.geoData.normalArrays.emplace_back(std::move(normals));
    assets.geoData.uvArrays.emplace_back(std::move(uvs));
    assets.geoData.indexArrays.emplace_back(std::move(indices));

    return mesh;
}

} // namespace madMJX