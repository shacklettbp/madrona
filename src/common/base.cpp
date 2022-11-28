#include <madrona/components.hpp>
#include <madrona/context.hpp>

namespace madrona {
using namespace math;

namespace base {

void registerTypes(ECSRegistry &registry)
{
    registry.registerComponent<Position>();
    registry.registerComponent<Rotation>();
}

}
}