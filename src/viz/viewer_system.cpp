#include "madrona/selector.hpp"
#include <madrona/viz/system.hpp>
#include <madrona/components.hpp>
#include <madrona/context.hpp>

#include <madrona/viz/interop.hpp>

namespace madrona::viz {
using namespace base;
using namespace math;

struct RenderableArchetype : public Archetype<
    InstanceData
> {};

struct VizCameraArchetype : public Archetype<
    PerspectiveCameraData
> {};

struct ViewerSystemState {
    uint32_t *totalNumViews;
    uint32_t *voxels;

    float aspectRatio;
};

struct RecordSystemState {
    bool *episodeDone;
};

inline void instanceTransformUpdate(Context &ctx,
                                    const Position &pos,
                                    const Rotation &rot,
                                    const Scale &scale,
                                    const ObjectID &obj_id,
                                    const Renderable &renderable)
{
    // Just update the instance data that is associated with this entity
    InstanceData &data = ctx.get<InstanceData>(renderable.renderEntity);
    data.position = pos;
    data.rotation = rot;
    data.scale = scale;
    data.worldIDX = ctx.worldID().idx;
    data.objectID = obj_id.idx;
}

uint32_t * VizRenderingSystem::getVoxelPtr(Context &ctx) {
    auto &sys_state = ctx.singleton<ViewerSystemState>();
    return sys_state.voxels;
}

inline void viewTransformUpdate(Context &ctx,
                                const Position &pos,
                                const Rotation &rot,
                                const VizCamera &viz_cam)
{
    Vector3 camera_pos = pos + viz_cam.cameraOffset;

    PerspectiveCameraData &cam_data = 
        ctx.get<PerspectiveCameraData>(viz_cam.cameraEntity);

    cam_data.position = Vector4::fromVector3(camera_pos, 1.0f);
    cam_data.rotation = rot.inv();
}

inline void exportCounts(Context &ctx,
                         ViewerSystemState &viewer_state)
{
#if 0
    *viewer_state.numInstances = viewer_state.numInstancesInternal;
    *viewer_state.numViews = viewer_state.numViewsInternal;

    viewer_state.numInstancesInternal = 0;
#endif

    if (ctx.worldID().idx == 0) {
#if defined(MADRONA_GPU_MODE)
        auto *state_mgr = mwGPU::getStateManager();
        *viewer_state.totalNumViews = state_mgr->getArchetypeNumRows<
            VizCameraArchetype>();
#else
    (void)viewer_state;
#endif
    }
}

void VizRenderingSystem::registerTypes(ECSRegistry &registry,
                                       const VizECSBridge *bridge)
{
    registry.registerComponent<VizCamera>();
    registry.registerComponent<Renderable>();
    registry.registerComponent<PerspectiveCameraData>();
    registry.registerComponent<InstanceData>();

    // Pointers get set in VizRenderingSystem::init
    registry.registerArchetype<VizCameraArchetype>(
        ComponentSelector<PerspectiveCameraData>(ComponentSelectImportPointer),
        ArchetypeImportOffsets);
    registry.registerArchetype<RenderableArchetype>(
        ComponentSelector<InstanceData>(ComponentSelectImportPointer),
        ArchetypeImportOffsets);

#if defined(MADRONA_GPU_MODE)
    auto *state_mgr = mwGPU::getStateManager();
    state_mgr->setArchetypeSortOffsets<RenderableArchetype>(
        bridge->instanceOffsets);
    state_mgr->setArchetypeComponent<RenderableArchetype, InstanceData>(
        bridge->instances);
    state_mgr->setArchetypeComponent<VizCameraArchetype, PerspectiveCameraData>(
        bridge->views);
#else
    (void)bridge;
#endif

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
        instanceTransformUpdate,
            Position,
            Rotation,
            Scale,
            ObjectID,
            Renderable
        >>(deps);

    auto viewdata_update = builder.addToGraph<ParallelForNode<Context,
        viewTransformUpdate,
            Position,
            Rotation,
            VizCamera
        >>({instance_setup});

    auto export_counts = builder.addToGraph<ParallelForNode<Context,
        exportCounts,
            ViewerSystemState
        >>({viewdata_update});

#ifdef MADRONA_GPU_MODE
    // Need to sort the instances, as well as the views
    auto sort_instances = 
        builder.addToGraph<SortArchetypeNode<RenderableArchetype, WorldID>>(
            {export_counts});

    auto post_instance_sort_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_instances});

    auto sort_views = 
        builder.addToGraph<SortArchetypeNode<VizCameraArchetype, WorldID>>(
            {post_instance_sort_reset_tmp});

    auto post_view_sort_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_views});

    return post_view_sort_reset_tmp;
#endif

    return export_counts;
}

void VizRenderingSystem::reset(Context &ctx)
{
    (void)ctx;
    // Nothing to reset
}

void VizRenderingSystem::init(Context &ctx,
                              const VizECSBridge *bridge)
{
    auto &system_state = ctx.singleton<ViewerSystemState>();

    int32_t world_idx = ctx.worldID().idx;

    system_state.totalNumViews = bridge->totalNumViews;
#if 0
    system_state.views = bridge->views[world_idx];
    system_state.numViews = &bridge->numViews[world_idx];
    system_state.instances = bridge->instances[world_idx];
    system_state.numInstances = &bridge->numInstances[world_idx];
#endif

    system_state.aspectRatio = 
        (float)bridge->renderWidth / (float)bridge->renderHeight;

    system_state.voxels = bridge->voxels;

    auto &record_state = ctx.singleton<RecordSystemState>();
    if (bridge->episodeDone != nullptr) {
        record_state.episodeDone = &bridge->episodeDone[world_idx];
    } else {
        record_state.episodeDone = nullptr;
    }
}

void VizRenderingSystem::makeEntityRenderable(Context &ctx,
                                              Entity e)
{
    Entity render_entity = ctx.makeEntity<RenderableArchetype>();
    ctx.get<Renderable>(e).renderEntity = render_entity;
}

void VizRenderingSystem::attachEntityToView(Context &ctx,
                                            Entity e,
                                            float vfov_degrees,
                                            float z_near,
                                            const math::Vector3 &camera_offset)
{
    Entity camera_entity = ctx.makeEntity<VizCameraArchetype>();
    ctx.get<VizCamera>(e) = { camera_entity, camera_offset };

    PerspectiveCameraData &cam_data = 
        ctx.get<PerspectiveCameraData>(camera_entity);

    auto &state = ctx.singleton<ViewerSystemState>();

    float fov_scale = tanf(toRadians(vfov_degrees * 0.5f));
    float x_scale = fov_scale / state.aspectRatio;
    float y_scale = -fov_scale;

    cam_data = PerspectiveCameraData {
        { /* Position */ }, 
        { /* Rotation */ }, 
        x_scale, y_scale, 
        z_near, 
        ctx.worldID().idx,
        0 // Padding
    };
}

void VizRenderingSystem::cleanupViewingEntity(Context &ctx,
                                              Entity e)
{
    Entity view_entity = ctx.get<VizCamera>(e).cameraEntity;
    ctx.destroyEntity(view_entity);
}

void VizRenderingSystem::cleanupRenderableEntity(Context &ctx,
                                                 Entity e)
{
    Entity render_entity = ctx.get<Renderable>(e).renderEntity;
    ctx.destroyEntity(render_entity);
}

void VizRenderingSystem::markEpisode(Context &ctx)
{
    auto &record_state = ctx.singleton<RecordSystemState>();
    if (record_state.episodeDone != nullptr) {
        *record_state.episodeDone = true;
    }
}

}
