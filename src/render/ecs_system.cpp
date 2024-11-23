#ifndef MADRONA_GPU_MODE
#include <bit>
#endif

#include <madrona/mesh_bvh.hpp>
#include <madrona/render/ecs.hpp>
#include <madrona/components.hpp>
#include <madrona/context.hpp>

#include "ecs_interop.hpp"

#ifdef MADRONA_GPU_MODE
#include <madrona/bvh.hpp>
#include <madrona/mw_gpu/const.hpp>
#include <madrona/mw_gpu/host_print.hpp>
#define LOG(...) mwGPU::HostPrint::log(__VA_ARGS__)
#else
#define LOG(...)
#endif

namespace madrona::render::RenderingSystem {
using namespace base;
using namespace math;

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

    MeshBVH *bvhs;
    uint32_t numBVHs;

    bool enableRaycaster;
};

static inline uint32_t leftShift3(uint32_t x)
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

static inline uint32_t encodeMorton3(const Vector3 &v)
{
#ifdef MADRONA_GPU_MODE
    uint32_t x_u32 = __float_as_uint(v.x);
    uint32_t y_u32 = __float_as_uint(v.y);
    uint32_t z_u32 = __float_as_uint(v.z);
#else
    uint32_t x_u32 = std::bit_cast<uint32_t>(v.x);
    uint32_t y_u32 = std::bit_cast<uint32_t>(v.y);
    uint32_t z_u32 = std::bit_cast<uint32_t>(v.z);
#endif

    // FIXME: not convinced this is correct

    return (leftShift3(z_u32) << 2) | 
           (leftShift3(y_u32) << 1) | 
            leftShift3(x_u32);
}

inline void mortonCodeUpdate(Context &ctx,
                             Entity e,
                             const Position &pos,
                             const Renderable &renderable)
{
    (void)e;

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

    auto &system_state = ctx.singleton<RenderingSystemState>();
#else
    (void)renderable;

    // Just update the instance data that is associated with this entity
    auto &system_state = ctx.singleton<RenderingSystemState>();

    if (!system_state.totalNumInstancesCPU) {
        return;
    }
    
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

    // Get the root AABB from the model and translate it to store
    // it in the TLBVHNode structure.

#ifdef MADRONA_GPU_MODE
    bool raycast_enabled = 
        mwGPU::GPUImplConsts::get().raycastOutputResolution != 0;

    if (raycast_enabled) {
        MeshBVH *bvh = (MeshBVH *)
            mwGPU::GPUImplConsts::get().meshBVHsAddr +
            obj_id.idx;

        math::AABB aabb = bvh->rootAABB.applyTRS(
                data.position, data.rotation, data.scale);

        ctx.get<TLBVHNode>(renderable.renderEntity).aabb = aabb;
    }
#endif
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

    ctx.get<RenderOutputIndex>(cam.cameraEntity).index = 
        ctx.loc(e).row;

#else
    auto &system_state = ctx.singleton<RenderingSystemState>();

    if (!system_state.totalNumViewsCPU) {
        return;
    }

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

#if MADRONA_GPU_MODE
    auto &gpu_consts = mwGPU::GPUImplConsts::get();

    BVHInternalData *bvh_internals =
        (BVHInternalData *)gpu_consts.bvhInternalData;

    if (bvh_internals != nullptr) {
        uint32_t num_views =
            state_mgr->getArchetypeNumRows<RenderCameraArchetype>();
        bvh_internals->numViews = num_views;
    }
#endif
}
#endif

void registerTypes(ECSRegistry &registry,
                   const RenderECSBridge *bridge)
{
#ifdef MADRONA_GPU_MODE
    uint32_t render_output_res = 
        mwGPU::GPUImplConsts::get().raycastOutputResolution;

    uint32_t rgb_output_bytes = render_output_res * render_output_res * 4;
    uint32_t depth_output_bytes = render_output_res * render_output_res * 4;

    // Make sure to have something there even if raycasting was disabled.
    if (depth_output_bytes == 0) {
        rgb_output_bytes = 4;
        depth_output_bytes = 4;
    } else if (mwGPU::GPUImplConsts::get().raycastRGBD == 0) {
        // Depth always renders whether we're in RGBD or Depth so we just 
        // disable RGB rendering.
        rgb_output_bytes = 4;
    }
#else
    uint32_t rgb_output_bytes = 4;
    uint32_t depth_output_bytes = 4;
#endif

    registry.registerComponent<RenderCamera>();
    registry.registerComponent<Renderable>();
    registry.registerComponent<PerspectiveCameraData>();
    registry.registerComponent<InstanceData>();
    registry.registerComponent<MortonCode>();

    registry.registerComponent<RGBOutputBuffer>(rgb_output_bytes);
    registry.registerComponent<DepthOutputBuffer>(depth_output_bytes);

    registry.registerComponent<RenderOutputIndex>();

    registry.registerComponent<RenderOutputRef>();
    registry.registerComponent<TLBVHNode>();

    registry.registerArchetype<RaycastOutputArchetype>();


    // Pointers get set in RenderingSystem::init
    if (bridge) {
#if defined(MADRONA_GPU_MODE)
        registry.registerArchetype<RenderCameraArchetype>(
            ComponentMetadataSelector<PerspectiveCameraData>(ComponentFlags::ImportMemory),
            ArchetypeFlags::None,
            bridge->maxViewsPerworld);
        registry.registerArchetype<RenderableArchetype>(
            ComponentMetadataSelector<InstanceData,TLBVHNode>(ComponentFlags::ImportMemory,ComponentFlags::ImportMemory),
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
        state_mgr->setArchetypeComponent<RenderableArchetype, TLBVHNode>(
            bridge->aabbs);
    }

#if 0
    auto *state_mgr = mwGPU::getStateManager();
    auto instance_ptr = (void *)state_mgr->getArchetypeComponent<
        RenderableArchetype, InstanceData>();

    printf("From rendering system init, instance_ptr=%p\n", instance_ptr);
#endif
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

    (void)mortoncode_update;

#ifdef MADRONA_GPU_MODE
    // Need to sort the instances, as well as the views

    // Need to sort by worlds first to handle deleted RenderableArchetypes
    auto sort_instances_by_world1 = 
        builder.addToGraph<SortArchetypeNode<RenderableArchetype, WorldID>>(
            {mortoncode_update});

    // Then sort by morton
    auto sort_instances_by_morton =
        builder.addToGraph<SortArchetypeNode<RenderableArchetype, MortonCode>>(
            {sort_instances_by_world1});

    // Then sort by world again to group up by world
    auto sort_instances_by_world2 = 
        builder.addToGraph<SortArchetypeNode<RenderableArchetype, WorldID>>(
            {sort_instances_by_morton});

    auto post_instance_sort_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_instances_by_world2});

#if 0
    auto sort_views = 
        builder.addToGraph<SortArchetypeNode<RenderCameraArchetype, WorldID>>(
            {post_instance_sort_reset_tmp});
#endif

    auto sort_views_world = builder.addToGraph<SortArchetypeNode<
        RenderCameraArchetype, WorldID>>(
                {post_instance_sort_reset_tmp});

    auto sort_views_index = builder.addToGraph<SortArchetypeNode<
        RenderCameraArchetype, RenderOutputIndex>>(
                {sort_views_world});

    auto post_view_sort_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_views_index});

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

    if (bridge) {
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

#if 0
    bool raycast_enabled = 
        mwGPU::GPUImplConsts::get().raycastOutputResolution != 0;

    system_state.enableRaycaster = raycast_enabled;
#endif
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
    ctx.get<RenderCamera>(e) = { 
        camera_entity, fov_scale, z_near, camera_offset 
    };

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

#ifdef MADRONA_GPU_MODE
    bool raycast_enabled = 
        mwGPU::GPUImplConsts::get().raycastOutputResolution != 0;
#else
    bool raycast_enabled = false;
#endif

    if (raycast_enabled) {
        Entity render_output_entity = ctx.makeEntity<RaycastOutputArchetype>();

        RenderOutputRef &ref = ctx.get<RenderOutputRef>(camera_entity);
        ref.outputEntity = render_output_entity;
    }
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

// Add this later when we decide to make the renderer more flexible
#if 0
void setEntityOutputIndex(Context &ctx, Entity e, uint32_t index)
{
    auto cam_entity = ctx.get<RenderCamera>(e).cameraEntity;
    ctx.get<RenderOutputIndex>(cam_entity).index = index;
}
#endif

}
