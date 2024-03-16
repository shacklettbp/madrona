#include <madrona/components.hpp>
#include <madrona/context.hpp>
#include <madrona/registry.hpp>

namespace madrona {
using namespace math;

namespace base {

void registerTypes(ECSRegistry &registry)
{
    registry.registerComponent<Position>();
    registry.registerComponent<Rotation>();
    registry.registerComponent<Scale>();
    registry.registerComponent<ObjectID>();

    registry.registerBundle<ObjectInstance>();
}

}
}
