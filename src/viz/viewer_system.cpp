#include <madrona/viz/system.hpp>
#include <madrona/components.hpp>

#include "interop.hpp"

namespace madrona::viz {
using namespace base;
using namespace math;

struct ViewerSystemState {
    PerspectiveCameraData *views;
    uint32_t *numViews;
    InstanceData *instances;
    uint32_t *numInstances;
    float aspectRatio;
};


inline void clearInstanceCount(Context &,
                               const ViewerRenderingSystem &sys_state)
{
    *(sys_state.numInstances) = 0;
}

inline void instanceTransformSetup(Context &ctx,
                                   const Position &pos,
                                   const Rotation &rot,
                                   const Scale &scale,
                                   const ObjectID &obj_id)
{
    ViewerSystemState &sys_state = ctx.singleton<ViewerSystemState>();

    AtomicU32Ref inst_count_atomic(*sys_state.numInstances);
    uint32_t inst_idx = inst_count_atomic.fetch_add_relaxed(1);

    sys_state.instances[inst_idx] = InstanceData {
        pos,
        rot,
        scale,
        obj_id.idx,
        0,
    };
}

void ViewerRenderingSystem::registerTypes(ECSRegistry &registry)
{
    registry.registerSingleton<ViewerSystemState>();
}

TaskGraph::NodeID ViewerRenderingSystem::setupTasks(
    TaskGraph::Builder &builder,
    Span<const TaskGraph::NodeID> deps)
{
    // FIXME: It feels like we should have persistent slots for renderer
    // state rather than needing to continually reset the instance count
    // and recreate the buffer. However, this might be hard to handle with
    // double buffering
    auto instance_clear = builder.addToGraph<ParallelForNode<Context,
        clearInstanceCount,
        RendererState>>(deps);

    auto instance_setup = builder.addToGraph<ParallelForNode<Context,
        instanceTransformSetup,
        Position,
        Rotation,
        Scale,
        ObjectID>>({instance_clear});

    auto viewdata_update = builder.addToGraph<ParallelForNode<Context,
        updateViewData,
        Position,
        Rotation,
        ViewSettings>>({instance_setup});

#ifdef MADRONA_GPU_MODE
    auto readback_count = builder.addToGraph<ParallelForNode<Context,
        readbackCount,
        RendererState>>({viewdata_update});

    return readback_count;
#else
    return viewdata_update;
}

void ViewerRenderingSystem::reset(Context &ctx)
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
