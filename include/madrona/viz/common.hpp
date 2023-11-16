#pragma once

#include <cstdint>
#include <madrona/math.hpp>

namespace madrona::viz {
    
// Required for rendering the viewer image
struct ViewerInput {
    // Which world to render in the flycam
    uint32_t worldIdx;

    // Camera world position/direction configuration
    math::Vector3 position;
    math::Vector3 forward;
    math::Vector3 up;
    math::Vector3 right;

    // Camera configuration
    bool usePerspective = true;
    float fov = 60.0f;
    float orthoHeight = 0.5f;
    math::Vector2 mousePrev = {0.0f, 0.0f};

    bool requestedScreenshot;
    const char *screenshotFilePath;
};

// Passed out after the flycam image has been rendered
struct ViewerFrame;

}

namespace madrona::render {

// Configures an individual light source
struct LightConfig {
    bool isDirectional;

    // Used for direction or position depending on value of isDirectional
    math::Vector3 dir;
    math::Vector3 color;
};

// If voxel generation is to happen
struct VoxelConfig {
    uint32_t xLength;
    uint32_t yLength;
    uint32_t zLength;
};

static inline float srgbToLinear(float srgb)
{
    if (srgb <= 0.04045f) {
        return srgb / 12.92f;
    }

    return powf((srgb + 0.055f) / 1.055f, 2.4f);
}

static inline math::Vector4 rgb8ToFloat(uint8_t r, uint8_t g, uint8_t b)
{
    return {
        srgbToLinear((float)r / 255.f),
        srgbToLinear((float)g / 255.f),
        srgbToLinear((float)b / 255.f),
        1.f,
    };
}

}
