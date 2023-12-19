#pragma once

#include <cstdint>
#include <madrona/math.hpp>

namespace madrona::viz {

enum class ViewerType : uint32_t {
    Flycam, Grid
};

struct ViewerCam {
    math::Vector3 position;
    math::Vector3 fwd;
    math::Vector3 up;
    math::Vector3 right;

    bool perspective = true;
    float fov = 60.f;
    float orthoHeight = 5.f;
    math::Vector2 mousePrev {0.f, 0.f};
};
    
// Required for rendering the viewer image
struct ViewerControl {
    // Which world to render in the flycam
    uint32_t worldIdx;
    uint32_t viewIdx;
    uint32_t controlIdx;
    bool linkViewControl;

    ViewerCam flyCam;

    bool requestedScreenshot;
    char screenshotFilePath[256];

    ViewerType viewerType;
    uint32_t gridWidth;
    uint32_t gridImageSize;
    float batchRenderOffsetX;
    float batchRenderOffsetY;
    bool batchRenderShowDepth;
};

}
