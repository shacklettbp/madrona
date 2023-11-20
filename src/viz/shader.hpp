#pragma once

namespace madrona::render {

namespace shader {

using float4x4 = madrona::math::Mat4x4;
using float3x3 = madrona::math::Mat3x3;
using float4 = madrona::math::Vector4;
using float3 = madrona::math::Vector3;
using float2 = madrona::math::Vector2;
using uint = uint32_t;
using uint2 = struct { uint32_t v[2]; };

#include "shaders/shader_common.h"

}

}
