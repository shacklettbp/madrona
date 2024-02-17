#include <madrona/render/ecs.hpp>
#include <madrona/components.hpp>
#include <madrona/context.hpp>

#include "ecs_interop.hpp"

namespace madrona::render::RenderingSystem {
using namespace base;
using namespace math;

struct RenderableArchetype : public Archetype<
    InstanceData,

    // For BVH support, we need to sort these not just be world ID,
    // but first by morton code too.
    MortonCode
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

inline uint32_t leftShift3(uint32_t x)
{
    if (x == (1 << 10)) {
        --x;
    }

    x = (x | (x << 16)) & 0b00000011000000000000000011111111;
    x = (x | (x << 8)) & 0b00000011000000001111000000001111;
    x = (x | (x << 4)) & 0b00000011000011000011000011000011;
    x = (x | (x << 2)) & 0b00001001001001001001001001001001;

    return x;
}

uint32_t encodeMorton3(const Vector3 &v) {
    return (leftShift3(*((uint32_t *)&v.z)) << 2) | 
           (leftShift3(*((uint32_t *)&v.y)) << 1) | 
            leftShift3(*((uint32_t *)&v.x));
}

inline void mortonCodeUpdate(Context &ctx,
                             Entity e,
                             const Position &pos,
                             const Renderable &renderable)
{
    // Calculate and set the morton code
    MortonCode &morton_code = ctx.get<MortonCode>(renderable.renderEntity);
    morton_code = encodeMorton3(pos);
}

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

uint32_t * getVoxelPtr(Context &ctx)
{
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

#ifdef MADRONA_GPU_MODE
inline void exportCountsGPU(Context &ctx,
                            RenderingSystemState &sys_state)
{
    // FIXME: Add option for global, across worlds, systems
    if (ctx.worldID().idx != 0) {
        return;
    }

    auto state_mgr = mwGPU::getStateManager();

    if (sys_state.totalNumViews) {
        *sys_state.totalNumViews = state_mgr->getArchetypeNumRows<
            RenderCameraArchetype>();
        *sys_state.totalNumInstances = state_mgr->getArchetypeNumRows<
            RenderableArchetype>();
    }

    uint32_t *morton_codes = state_mgr->getArchetypeComponent<
        RenderableArchetype, MortonCode>();
    
    WorldID *world_ids = state_mgr->getArchetypeComponent<
        RenderableArchetype, WorldID>();

    for (int i = 0; 
         i < state_mgr->getArchetypeNumRows<RenderableArchetype>();
         ++i) {
        printf("%d: %u\n", world_ids[i].idx, morton_codes[i]);
    }
}
#endif

void registerTypes(ECSRegistry &registry,
                   const RenderECSBridge *bridge)
{
    registry.registerComponent<RenderCamera>();
    registry.registerComponent<Renderable>();
    registry.registerComponent<PerspectiveCameraData>();
    registry.registerComponent<InstanceData>();
    registry.registerComponent<MortonCode>();


    // Pointers get set in RenderingSystem::init
    if (bridge) {
#if defined(MADRONA_GPU_MODE)
        registry.registerArchetype<RenderCameraArchetype>(
            ComponentMetadataSelector<PerspectiveCameraData>(ComponentFlags::ImportMemory),
            ArchetypeFlags::None,
            bridge->maxViewsPerworld);
        registry.registerArchetype<RenderableArchetype>(
            ComponentMetadataSelector<InstanceData>(ComponentFlags::ImportMemory),
            ArchetypeFlags::ImportOffsets,
            bridge->maxInstancesPerWorld);
#else
        registry.registerArchetype<RenderCameraArchetype>();
        registry.registerArchetype<RenderableArchetype>();
#endif
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
}

TaskGraphNodeID setupTasks(TaskGraphBuilder &builder,
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

    auto mortoncode_update = builder.addToGraph<ParallelForNode<Context,
         mortonCodeUpdate,
            Entity,
            Position,
            Renderable
        >>({viewdata_update});

#ifdef MADRONA_GPU_MODE
    // Need to sort the instances, as well as the views
    auto sort_instances_by_morton =
        builder.addToGraph<SortArchetypeNode<RenderableArchetype, MortonCode>>(
            {mortoncode_update});

    auto sort_instances_by_world = 
        builder.addToGraph<SortArchetypeNode<RenderableArchetype, WorldID>>(
            {sort_instances_by_morton});

    auto post_instance_sort_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_instances_by_world});

    auto sort_views = 
        builder.addToGraph<SortArchetypeNode<RenderCameraArchetype, WorldID>>(
            {post_instance_sort_reset_tmp});

    auto post_view_sort_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_views});

    auto export_counts = builder.addToGraph<ParallelForNode<Context,
        exportCountsGPU,
            RenderingSystemState
        >>({post_view_sort_reset_tmp});

    return export_counts;
#else
    return viewdata_update;
#endif
}

void init(Context &ctx,
          const RenderECSBridge *bridge)
{
    auto &system_state = ctx.singleton<RenderingSystemState>();

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
}

void makeEntityRenderable(Context &ctx,
                          Entity e)
{
    Entity render_entity = ctx.makeEntity<RenderableArchetype>();
    ctx.get<Renderable>(e).renderEntity = render_entity;
}

void attachEntityToView(Context &ctx,
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

void cleanupViewingEntity(Context &ctx,
                          Entity e)
{
    Entity view_entity = ctx.get<RenderCamera>(e).cameraEntity;
    ctx.destroyEntity(view_entity);
}

void cleanupRenderableEntity(Context &ctx,
                             Entity e)
{
    Entity render_entity = ctx.get<Renderable>(e).renderEntity;
    ctx.destroyEntity(render_entity);
}

}
