#include <madrona/render.hpp>
#include <madrona/components.hpp>
#include <madrona/context.hpp>

namespace madrona {
using namespace base;
using namespace math;

namespace render {

void RenderSetupSystem::registerTypes(ECSRegistry &registry)
{
    registry.registerComponent<ObjectToWorld>();
    registry.registerComponent<ObjectID>();
    registry.registerComponent<RenderEntity>();

    registry.registerArchetype<RenderObject>();
}

inline void renderSetup(Context &ctx,
                        const Position &pos,
                        const Rotation &rot,
                        const RenderEntity &render_e)
{
    auto &o2w = ctx.getUnsafe<ObjectToWorld>(render_e.renderEntity);
    o2w = Mat3x4::fromTRS(pos, rot);
}

TaskGraph::NodeID RenderSetupSystem::setupTasks(TaskGraph::Builder &builder,
    Span<const TaskGraph::NodeID> deps)
{
    return builder.parallelForNode<Context,
                                   renderSetup,
                                   Position,
                                   Rotation,
                                   RenderEntity>(deps);
}

}
}
