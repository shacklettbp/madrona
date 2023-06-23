#pragma once

#include <madrona/math.hpp>
#include <madrona/state.hpp>

namespace madrona::render {

struct ViewID {
    int32_t idx;
};

struct ViewSettings {
    float xScale;
    float yScale;
    float zNear;
    math::Vector3 cameraOffset;
    ViewID viewID;
};

void registerMultiViewTypes(ECSRegistry &registry);

}
