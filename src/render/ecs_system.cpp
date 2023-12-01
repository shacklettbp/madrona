#include <madrona/render/ecs.hpp>
#include <madrona/components.hpp>
#include <madrona/context.hpp>

#include "ecs_interop.hpp"

namespace madrona::render {
using namespace base;
using namespace math;

struct RenderableArchetype : public Archetype<
    InstanceData
> {};

struct RenderCameraArchetype : public Archetype<
    PerspectiveCameraData
> {};

struct RenderingSystemState {
    uint32_t *totalNumViews;
    uint32_t *totalNumInstances;

    uint32_t *voxels;
    float aspectRatio;

    // This is used if on the CPU backend
    AtomicU32 *totalNumViewsCPU;
    AtomicU32 *totalNumInstancesCPU;

    // This is used if on the CPU backend
    InstanceData *instancesCPU;
    PerspectiveCameraData *viewsCPU;

    // World IDs (keys used in the key-value sorting)
    // Also only used when on the CPU backend
    uint64_t *instanceWorldIDsCPU;
    uint64_t *viewWorldIDsCPU;
};

struct RecordSystemState {
    bool *episodeDone;
};

inline void instanceTransformUpdate(Context &ctx,
                                    Entity e,
                                    const Position &pos,
                                    const Rotation &rot,
                                    const Scale &scale,
                                    const ObjectID &obj_id,
                                    const Renderable &renderable)
{
    // Just update the instance data that is associated with this entity
#if defined(MADRONA_GPU_MODE)
    (void)e;

    InstanceData &data = ctx.get<InstanceData>(renderable.renderEntity);
#else
    (void)renderable;

    // Just update the instance data that is associated with this entity
    auto &system_state = ctx.singleton<RenderingSystemState>();
    uint32_t instance_id = system_state.totalNumInstancesCPU->fetch_add<sync::acq_rel>(1);

    // Required for stable sorting on CPU
    system_state.instanceWorldIDsCPU[instance_id] = 
        ((uint64_t)ctx.worldID().idx << 32) | (uint64_t)e.id;

    InstanceData &data = system_state.instancesCPU[instance_id];
#endif

    data.position = pos;
    data.rotation = rot;
    data.scale = scale;
    data.worldIDX = ctx.worldID().idx;
    data.objectID = obj_id.idx;
}

uint32_t * RenderingSystem::getVoxelPtr(Context &ctx) {
    auto &sys_state = ctx.singleton<RenderingSystemState>();
    return sys_state.voxels;
}

inline void viewTransformUpdate(Context &ctx,
                                Entity e,
                                const Position &pos,
                                const Rotation &rot,
                                const RenderCamera &cam)
{
    Vector3 camera_pos = pos + cam.cameraOffset;

#if defined(MADRONA_GPU_MODE)
    (void)e;

    PerspectiveCameraData &cam_data = 
        ctx.get<PerspectiveCameraData>(cam.cameraEntity);
#else
    auto &system_state = ctx.singleton<RenderingSystemState>();
    uint32_t view_id = system_state.totalNumViewsCPU->fetch_add<sync::acq_rel>(1);

    // Required for stable sorting on CPU
    system_state.viewWorldIDsCPU[view_id] = 
        ((uint64_t)ctx.worldID().idx << 32) | (uint64_t)e.id;

    PerspectiveCameraData &cam_data = system_state.viewsCPU[view_id];
#endif
    cam_data.position = camera_pos;
    cam_data.rotation = rot.inv();
    cam_data.worldIDX = ctx.worldID().idx;

#if !defined(MADRONA_GPU_MODE)
    float x_scale = cam.fovScale / system_state.aspectRatio;
    float y_scale = -cam.fovScale;

    cam_data.xScale = x_scale;
    cam_data.yScale = y_scale;
    cam_data.zNear = cam.zNear;
#endif
}

inline void exportCounts(Context &ctx,
                         RenderingSystemState &viewer_state)
{
    (void)viewer_state;

    if (ctx.worldID().idx == 0) {
#if defined(MADRONA_GPU_MODE)
        auto *state_mgr = mwGPU::getStateManager();
        *viewer_state.totalNumViews = state_mgr->getArchetypeNumRows<
            RenderCameraArchetype>();
        *viewer_state.totalNumInstances = state_mgr->getArchetypeNumRows<
            RenderableArchetype>();
#else

#if 0
        *viewer_state.totalNumViews = viewer_state.totalNumViewsCPU->load_relaxed();
        *viewer_state.totalNumInstances = viewer_state.totalNumInstancesCPU->load_relaxed();

        // Reset the atomic counters
        viewer_state.totalNumViewsCPU->store_relaxed(0);
        viewer_state.totalNumInstancesCPU->store_relaxed(0);
#endif

#endif
    }
}

void RenderingSystem::registerTypes(ECSRegistry &registry,
                                       const RenderECSBridge *bridge)
{
    registry.registerComponent<RenderCamera>();
    registry.registerComponent<Renderable>();
    registry.registerComponent<PerspectiveCameraData>();
    registry.registerComponent<InstanceData>();


    // Pointers get set in RenderingSystem::init
    if (bridge) {
        registry.registerArchetype<RenderCameraArchetype>(
            ComponentMetadataSelector<PerspectiveCameraData>(ComponentFlags::ImportMemory),
            ArchetypeFlags::None,
            bridge->maxViewsPerworld);
        registry.registerArchetype<RenderableArchetype>(
            ComponentMetadataSelector<InstanceData>(ComponentFlags::ImportMemory),
            ArchetypeFlags::ImportOffsets,
            bridge->maxInstancesPerWorld);
    } else {
        registry.registerArchetype<RenderCameraArchetype>();
        registry.registerArchetype<RenderableArchetype>();
    }


#if defined(MADRONA_GPU_MODE)
    if (bridge) {
        auto *state_mgr = mwGPU::getStateManager();

        state_mgr->setArchetypeWorldOffsets<RenderableArchetype>(
            bridge->instanceOffsets);
        state_mgr->setArchetypeWorldOffsets<RenderCameraArchetype>(
            bridge->viewOffsets);

        state_mgr->setArchetypeComponent<RenderableArchetype, InstanceData>(
            bridge->instances);
        state_mgr->setArchetypeComponent<RenderCameraArchetype, PerspectiveCameraData>(
            bridge->views);
    }
#else
    (void)bridge;
#endif

    registry.registerSingleton<RenderingSystemState>();

    // Technically this singleton is only used in record mode
    registry.registerSingleton<RecordSystemState>();
}

TaskGraphNodeID RenderingSystem::setupTasks(
    TaskGraphBuilder &builder,
    Span<const TaskGraphNodeID> deps)
{
    // FIXME: It feels like we should have persistent slots for renderer
    // state rather than needing to continually reset the instance count
    // and recreate the buffer. However, this might be hard to handle with
    // double buffering
    auto instance_setup = builder.addToGraph<ParallelForNode<Context,
        instanceTransformUpdate,
            Entity,
            Position,
            Rotation,
            Scale,
            ObjectID,
            Renderable
        >>(deps);

    auto viewdata_update = builder.addToGraph<ParallelForNode<Context,
        viewTransformUpdate,
            Entity,
            Position,
            Rotation,
            RenderCamera
        >>({instance_setup});

    auto export_counts = builder.addToGraph<ParallelForNode<Context,
        exportCounts,
            RenderingSystemState
        >>({viewdata_update});

#ifdef MADRONA_GPU_MODE
    // Need to sort the instances, as well as the views
    auto sort_instances = 
        builder.addToGraph<SortArchetypeNode<RenderableArchetype, WorldID>>(
            {export_counts});

    auto post_instance_sort_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_instances});

    auto sort_views = 
        builder.addToGraph<SortArchetypeNode<RenderCameraArchetype, WorldID>>(
            {post_instance_sort_reset_tmp});

    auto post_view_sort_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_views});

    return post_view_sort_reset_tmp;
#endif

    return export_counts;
}

void RenderingSystem::reset(Context &ctx)
{
    (void)ctx;
    // Nothing to reset
}

void RenderingSystem::init(Context &ctx,
                              const RenderECSBridge *bridge)
{
    auto &system_state = ctx.singleton<RenderingSystemState>();

    int32_t world_idx = ctx.worldID().idx;

    // This is where the renderer will read out the totals
    system_state.totalNumViews = bridge->totalNumViews;
    system_state.totalNumInstances = bridge->totalNumInstances;

#if !defined(MADRONA_GPU_MODE)
    // This is just an atomic counter (final value will be moved to
    // the totalNumViews/Instances variables).
    system_state.totalNumViewsCPU = bridge->totalNumViewsCPUInc;
    system_state.totalNumInstancesCPU = bridge->totalNumInstancesCPUInc;

    // This is only relevant for the CPU backend
    system_state.instancesCPU = bridge->instances;
    system_state.viewsCPU = bridge->views;

    system_state.instanceWorldIDsCPU = bridge->instancesWorldIDs;
    system_state.viewWorldIDsCPU = bridge->viewsWorldIDs;
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

void RenderingSystem::makeEntityRenderable(Context &ctx,
                                              Entity e)
{
    Entity render_entity = ctx.makeEntity<RenderableArchetype>();
    ctx.get<Renderable>(e).renderEntity = render_entity;
}

void RenderingSystem::attachEntityToView(Context &ctx,
                                            Entity e,
                                            float vfov_degrees,
                                            float z_near,
                                            const math::Vector3 &camera_offset)
{
    float fov_scale = 1.0f / tanf(toRadians(vfov_degrees * 0.5f));

    Entity camera_entity = ctx.makeEntity<RenderCameraArchetype>();
    ctx.get<RenderCamera>(e) = { camera_entity, fov_scale, z_near, camera_offset };

    PerspectiveCameraData &cam_data = 
        ctx.get<PerspectiveCameraData>(camera_entity);

    auto &state = ctx.singleton<RenderingSystemState>();

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

void RenderingSystem::cleanupViewingEntity(Context &ctx,
                                           Entity e)
{
    Entity view_entity = ctx.get<RenderCamera>(e).cameraEntity;
    ctx.destroyEntity(view_entity);
}

void RenderingSystem::cleanupRenderableEntity(Context &ctx,
                                              Entity e)
{
    Entity render_entity = ctx.get<Renderable>(e).renderEntity;
    ctx.destroyEntity(render_entity);
}

void RenderingSystem::markEpisode(Context &ctx)
{
    auto &record_state = ctx.singleton<RecordSystemState>();
    if (record_state.episodeDone != nullptr) {
        *record_state.episodeDone = true;
    }
}

}
