#include <madrona/viz/system.hpp>
#include <madrona/components.hpp>
#include <madrona/context.hpp>

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
    uint32_t numInstancesInternal;
    uint32_t numViewsInternal;
    uint32_t *voxels;
};

struct RecordSystemState {
    bool *episodeDone;
};

inline void instanceTransformSetup(Context &ctx,
                                   const Position &pos,
                                   const Rotation &rot,
                                   const Scale &scale,
                                   const ObjectID &obj_id)
{
    auto &sys_state = ctx.singleton<ViewerSystemState>();

    AtomicU32Ref inst_count_atomic(sys_state.numInstancesInternal);
    uint32_t inst_idx = inst_count_atomic.fetch_add_relaxed(1);

    sys_state.instances[inst_idx] = InstanceData {
        pos,
        rot,
        scale,
        obj_id.idx,
        0,
    };
}

uint32_t * VizRenderingSystem::getVoxelPtr(Context &ctx) {
    auto &sys_state = ctx.singleton<ViewerSystemState>();
    return sys_state.voxels;
}


inline void updateViewData(Context &ctx,
                           const Position &pos,
                           const Rotation &rot,
                           const VizCamera &viz_cam)
{
    auto &sys_state = ctx.singleton<ViewerSystemState>();
    int32_t view_idx = viz_cam.viewIDX;

    Vector3 camera_pos = pos + viz_cam.cameraOffset;

    sys_state.views[view_idx] = PerspectiveCameraData {
        camera_pos,
        rot.inv(),
        viz_cam.xScale,
        viz_cam.yScale,
        viz_cam.zNear,
        {},
    };
}

inline void exportCounts(Context &,
                         ViewerSystemState &viewer_state)
{
    *viewer_state.numInstances = viewer_state.numInstancesInternal;
    *viewer_state.numViews = viewer_state.numViewsInternal;

    viewer_state.numInstancesInternal = 0;
}

void VizRenderingSystem::registerTypes(ECSRegistry &registry)
{
    registry.registerComponent<VizCamera>();
    registry.registerSingleton<ViewerSystemState>();

    // Technically this singleton is only used in record mode
    registry.registerSingleton<RecordSystemState>();
}

TaskGraphNodeID VizRenderingSystem::setupTasks(
    TaskGraphBuilder &builder,
    Span<const TaskGraphNodeID> deps)
{
    // FIXME: It feels like we should have persistent slots for renderer
    // state rather than needing to continually reset the instance count
    // and recreate the buffer. However, this might be hard to handle with
    // double buffering
    auto instance_setup = builder.addToGraph<ParallelForNode<Context,
        instanceTransformSetup,
            Position,
            Rotation,
            Scale,
            ObjectID
        >>(deps);

    auto viewdata_update = builder.addToGraph<ParallelForNode<Context,
        updateViewData,
            Position,
            Rotation,
            VizCamera
        >>({instance_setup});

    auto export_counts = builder.addToGraph<ParallelForNode<Context,
        exportCounts,
            ViewerSystemState
        >>({viewdata_update});

    return export_counts;
}

void VizRenderingSystem::reset(Context &ctx)
{
    auto &system_state = ctx.singleton<ViewerSystemState>();
    system_state.numViewsInternal = 0;
}

void VizRenderingSystem::init(Context &ctx,
                              const VizECSBridge *bridge)
{
    auto &system_state = ctx.singleton<ViewerSystemState>();

    int32_t world_idx = ctx.worldID().idx;

    system_state.views = bridge->views[world_idx];
    system_state.numViews = &bridge->numViews[world_idx];
    system_state.instances = bridge->instances[world_idx];
    system_state.numInstances = &bridge->numInstances[world_idx];
    system_state.aspectRatio = 
        (float)bridge->renderWidth / (float)bridge->renderHeight;

    system_state.voxels = bridge->voxels;

    auto &record_state = ctx.singleton<RecordSystemState>();
    if (bridge->episodeDone != nullptr) {
        record_state.episodeDone = &bridge->episodeDone[world_idx];
    } else {
        record_state.episodeDone = nullptr;
    }

    system_state.numInstancesInternal = 0;
    system_state.numViewsInternal = 0;
}

VizCamera VizRenderingSystem::setupView(
    Context &ctx,
    float vfov_degrees,
    float z_near,
    math::Vector3 camera_offset,
    int32_t view_idx)
{
    auto &sys_state = ctx.singleton<ViewerSystemState>();

    float fov_scale = tanf(toRadians(vfov_degrees * 0.5f));

    sys_state.numViewsInternal += 1;

    float x_scale = fov_scale / sys_state.aspectRatio;
    float y_scale = -fov_scale;

    return VizCamera {
        x_scale,
        y_scale,
        z_near,
        camera_offset,
        view_idx,
    };
}

void VizRenderingSystem::markEpisode(Context &ctx)
{
    auto &record_state = ctx.singleton<RecordSystemState>();
    if (record_state.episodeDone != nullptr) {
        *record_state.episodeDone = true;
    }
}

}
