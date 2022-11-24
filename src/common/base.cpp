#include <madrona/components.hpp>
#include <madrona/context.hpp>

namespace madrona {
namespace base {

void registerECS(Context &ctx)
{
    ctx.registerComponent<Position>();
    ctx.registerComponent<Rotation>();
}

}
}
