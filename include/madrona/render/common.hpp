#pragma once

#include <madrona/math.hpp>

namespace madrona::render {

struct APILib {};
struct APIBackend {};
struct GPUDevice {};

// If voxel generation is to happen
struct VoxelConfig {
    uint32_t xLength;
    uint32_t yLength;
    uint32_t zLength;
};

inline float srgbToLinear(float srgb);
inline math::Vector4 srgb8ToFloat(uint8_t r, uint8_t g, uint8_t b);

}

#include "common.inl"
