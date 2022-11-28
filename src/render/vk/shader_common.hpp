#pragma once
#include <cstdint>
#include <madrona/math.hpp>

namespace madrona {
namespace render {

namespace shader {
using float3 = madrona::math::Vector3;
using float2 = madrona::math::Vector2;
#include "shader_common.h"
}

using shader::Vertex;
using shader::Mesh;
using shader::Object;

}
}
