#include <madrona/render/mw/components.hpp>

namespace madrona::render {

void registerMultiViewTypes(ECSRegistry &registry)
{
    registry.registerComponent<ViewSettings>();
}

}
