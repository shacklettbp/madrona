#include <madrona/viz/system.hpp>

#include "interop.hpp"

namespace madrona::viz {

struct ViewerSystemState {
    PerspectiveCameraData *views;
    uint32_t *numViews;
    InstanceData *instances;
    uint32_t *numInstances;
    float aspectRatio;
};

void ViewerRenderingSystem::registerTypes(ECSRegistry &registry)
{
    registry.registerSingleton<ViewerSystemState>();
}

TaskGraph::NodeID ViewerRenderingSystem::setupTasks(
    TaskGraph::Builder &builder,
    Span<const TaskGraph::NodeID> deps)
{
}

void ViewerRenderingSystem::init(Context &ctx,
                                 const ViewerECSBridge *bridge)
{
    auto &system_state = ctx.singleton<ViewerSystemState>();

    int32_t world_idx = ctx.worldID().idx;

    system_state.views = bridge->views[world_idx];
    system_state.numViews = &bridge->numViews[world_idx];
    system_state.instances = bridge->instances[world_idx];
    system_state.numInstances = &bridge->numInstances[world_idx];
    system_state.aspectRatio = 
        (float)bridge->renderWidth / (float)bridge->renderHeight;
}



}
