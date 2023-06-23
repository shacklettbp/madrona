#include <madrona/render/mw/components.hpp>

namespace madrona::render {

void MultiView::registerTypes(ECSRegistry &registry)
{
    registry.registerComponent<ViewSettings>();
}

ViewSettings MultiView::setupView(float vfov_degrees,
                                  float z_near,
                                  math::Vector3 camera_offset,
                                  ViewID view_id)
{
    float fov_scale = tanf(toRadians(vfov_degrees * 0.5f));

    float x_scale = fov_scale / aspect_ratio;
    float y_scale = fov_scale;
}

}
