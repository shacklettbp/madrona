#pragma once

#include <madrona/math.hpp>
#include <Metal/Metal.hpp>

namespace madrona::render::metal {

namespace shader {

using float2 = math::Vector2;
using float3 = math::Vector3;
using float4 = math::Vector4;
using packed_float2 = math::Vector2;
using packed_float3 = math::Vector3;
using packed_float4 = math::Vector4;

using float3x3 = math::Mat3x3;
using float4x3 = math::Mat3x4;
using float4x4 = math::Mat4x4;

#include "shaders/shader_common.h"
}

using Vertex = shader::Vertex;
using PackedVertex = shader::PackedVertex;
using MeshData = shader::MeshData;
using ObjectData = shader::ObjectData;
using DrawInstanceData = shader::DrawInstanceData;
using AssetsArgBuffer = shader::AssetsArgBuffer;
using DrawICBArgBuffer = shader::DrawICBArgBuffer;
using RenderArgBuffer = shader::RenderArgBuffer;

}
