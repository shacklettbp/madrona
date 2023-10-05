#pragma once

#include <madrona/math.hpp>

namespace madrona::render::vk {

namespace shader {

using float4 = madrona::math::Vector4;
using float3 = madrona::math::Vector3;
using float2 = madrona::math::Vector2;
using uint = unsigned int;
#include "shaders/shader_common.h"

}

}
