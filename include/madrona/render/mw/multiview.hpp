#pragma once

#include <madrona/math.hpp>
#include <madrona/context.hpp>

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

struct MultiView {
    static void registerTypes(ECSRegistry &registry);

    static ViewSettings setupView(float vfov_degrees,
                                  float z_near,
                                  math::Vector3 camera_offset,
                                  ViewID view_id);
};

}
