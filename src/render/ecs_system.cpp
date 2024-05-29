#ifdef MADRONA_GPU_MODE

#include <madrona/mesh_bvh.hpp>
#include <madrona/render/ecs.hpp>
#include <madrona/components.hpp>
#include <madrona/context.hpp>

#include "ecs_interop.hpp"

#ifdef MADRONA_GPU_MODE
#include <madrona/mw_gpu/const.hpp>
#define LOG(...) mwGPU::HostPrint::log(__VA_ARGS__)
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

    AtomicI32 numDeleted;

    // Specific to the script bots backend.
    struct {
        AtomicU32 totalNumInstances;

        InstanceData *instances;

        bool isTracking;

        uint32_t *exportedWorldID;
    } scriptBots;
};

uint32_t part1By1(uint32_t x)
{
    x &= 0x0000ffff;
    x = (x ^ (x <<  8)) & 0x00ff00ff;
    x = (x ^ (x <<  4)) & 0x0f0f0f0f;
    x = (x ^ (x <<  2)) & 0x33333333;
    x = (x ^ (x <<  1)) & 0x55555555;
    return x;
}

uint32_t encodeMorton2(uint32_t x, uint32_t y)
{
    return (part1By1(y) << 1) + part1By1(x);
}

uint32_t mortonVector2(const Vector2 &v)
{
    return encodeMorton2(*((uint32_t *)&v.x),
                         *((uint32_t *)&v.y));
}

// For script bots, we're dealing with 2D
inline void mortonCodeUpdate(Context &ctx,
                             Entity e,
                             const Position &pos,
                             const Renderable &renderable)
{
    (void)e;

    // Calculate and set the morton code
    MortonCode &morton_code = ctx.get<MortonCode>(renderable.renderEntity);
    morton_code = mortonVector2({pos.x, pos.y});
}

inline void instanceTransformUpdate(Context &ctx,
                                    Entity e,
                                    const Position &pos,
                                    const Rotation &rot,
                                    const Scale &scale,
                                    const ObjectID &obj_id,
                                    const Renderable &renderable)
{
    InstanceData &data = ctx.get<InstanceData>(renderable.renderEntity);

    data.position = { pos.x, pos.y };
    data.scale = { scale.d0, scale.d1 };
    data.objectIDX = obj_id.idx;
    data.worldIDX = ctx.worldID().idx;
    data.owner = e;
    data.zOffset = pos.z;

    Vector3 rotated_x = rot.rotateVec(Vector3{ 1.f, 0.f, 0.f });

    // Set the orientation
    data.viewDirPolar = atan2f(rotated_x.y, rotated_x.x);

    // For script bots, we don't read the BVH from the MeshBVH.
    // It's just a Box surrounding the circle of the bot.
    math::AABB model_space_aabb;
    if (obj_id.idx == 0) {
        // Object 0 is the agent
        data.viewDirPolar += -math::pi * 0.5f;

        model_space_aabb = {
            .pMin = { -1.f, -1.f, -1.f },
            .pMax = { +1.f, +1.f, +1.f }
        };
    } else {
        model_space_aabb = {
            .pMin = { -1.f, -1.f, 0.f },
            .pMax = { +1.f, +1.f, 0.f }
        };
    }

    math::AABB aabb = model_space_aabb.applyTRS(
            pos, rot, scale);

    ctx.get<TLBVHNode>(renderable.renderEntity).aabb = aabb;

    auto &state = ctx.singleton<RenderingSystemState>();

    if (state.scriptBots.isTracking) {
        // Only do this for world 0 for now.
        if (ctx.worldID().idx == *state.scriptBots.exportedWorldID) {
            uint32_t offset =
                state.scriptBots.totalNumInstances.fetch_add_relaxed(1);

            state.scriptBots.instances[offset] = data;
        }
    }
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
    Loc loc = ctx.loc(e);



    uint32_t render_output_res = 
        mwGPU::GPUImplConsts::get().raycastOutputResolution;


    Vector3 camera_pos = pos + cam.cameraOffset;


    PerspectiveCameraData &cam_data = 
        ctx.get<PerspectiveCameraData>(cam.cameraEntity);

    Vector3 rotated_x = rot.rotateVec(Vector3{ 1.f, 0.f, 0.f });



    // Set all the camera data information
    cam_data.position = {camera_pos.x, camera_pos.y};
    cam_data.viewDirPolar = atan2f(rotated_x.y, rotated_x.x);
    cam_data.numForwardRays = 3 * render_output_res / 4;
    cam_data.numBackwardRays = render_output_res / 4;
    cam_data.worldIDX = ctx.worldID().idx;
    // cam_data.rowIDX = loc.row;
    cam_data.zOffset = camera_pos.z;
}

#ifdef MADRONA_GPU_MODE
inline void exportCountsGPU(Context &ctx,
                            RenderingSystemState &sys_state)
{
    if (sys_state.scriptBots.isTracking && 
            ctx.worldID().idx == *sys_state.scriptBots.exportedWorldID) {
        uint32_t num_instances =
            sys_state.scriptBots.totalNumInstances.load_relaxed();

        // This is the number that the renderer will see.
        *sys_state.totalNumInstances = num_instances;

        *sys_state.totalNumViews = 0;

        sys_state.scriptBots.totalNumInstances.store_relaxed(0);
    }
}
#endif

void registerTypes(ECSRegistry &registry,
                   const RenderECSBridge *bridge)
{
#ifdef MADRONA_GPU_MODE
    uint32_t render_output_res = 
        mwGPU::GPUImplConsts::get().raycastOutputResolution;

    // The raycast output resolution is simply the number of pixels
    // for script bots because the visualization is just 1D
    uint32_t render_output_bytes = render_output_res;

    // Make sure to have something there even if raycasting was disabled.
    if (render_output_bytes == 0) {
        render_output_bytes = 4;
    }
#else
    uint32_t render_output_bytes = 4;
#endif

    registry.registerComponent<RenderCamera>();
    registry.registerComponent<Renderable>();
    registry.registerComponent<PerspectiveCameraData>();
    registry.registerComponent<InstanceData>();
    registry.registerComponent<MortonCode>();
    registry.registerComponent<RenderOutputBuffer>(render_output_bytes);

    // This is enough to store a number denoting the 
    registry.registerComponent<FinderOutputBuffer>(sizeof(FinderOutput));

    registry.registerComponent<RenderOutputRef>();
    registry.registerComponent<TLBVHNode>();

    registry.registerArchetype<RaycastOutputArchetype>();

    registry.registerSingleton<RenderingSystemState>();

    registry.registerArchetype<RenderCameraArchetype>();
    registry.registerArchetype<RenderableArchetype>();

#if 0
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
#endif

#endif
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

    system_state.numDeleted.store_relaxed(0);

    if (bridge) {
        system_state.totalNumViews = bridge->totalNumViews;
        system_state.totalNumInstances = bridge->totalNumInstances;

        system_state.scriptBots.totalNumInstances.store_relaxed(0);
        system_state.scriptBots.instances = bridge->instances;

        system_state.scriptBots.exportedWorldID = bridge->exportedWorldID;

        system_state.scriptBots.isTracking = true;
    } else {
        system_state.scriptBots.isTracking = false;
    }

#if 0
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

    cam_data = PerspectiveCameraData {
        { /* Position */ }, 
        0.f,
        0, 0,
        ctx.worldID().idx,
    };

#ifdef MADRONA_GPU_MODE
    bool raycast_enabled = 
        mwGPU::GPUImplConsts::get().raycastOutputResolution != 0;
#else
    bool raycast_enabled = false;
#endif

    if (raycast_enabled) {
        // Here, we aren't really treating these guys as entities. We are more
        // so just using the ECS as memory allocator.
        // So only create a new entity (i.e., allocate more space) if we need to.

        if (state.numDeleted.load_relaxed() == 0) {
            Entity render_output_entity = 
                ctx.makeEntity<RaycastOutputArchetype>();
        } else {
            state.numDeleted.fetch_add_relaxed(-1);
        }
    }
}

void cleanupViewingEntity(Context &ctx,
                          Entity e)
{
    auto &state = ctx.singleton<RenderingSystemState>();
    state.numDeleted.fetch_add_relaxed(1);

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

#endif
