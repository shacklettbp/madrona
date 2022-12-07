#pragma once

#include <madrona/math.hpp>
#include <madrona/span.hpp>

namespace madrona {
namespace render {

struct SourceVertex {
    math::Vector3 position;
    math::Vector3 normal;
    math::Vector4 tangentAndSign;
    math::Vector2 uv;
};

struct SourceMesh {
    Span<const SourceVertex> vertices;
    Span<const uint32_t> indices;
};

struct SourceObject {
    Span<const SourceMesh> meshes;
};


}
}
