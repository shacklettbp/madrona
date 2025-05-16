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
    uint32_t *totalNumLights;
    uint32_t *voxels;
    float aspectRatio;

    // This is used if on the CPU backend
    AtomicU32 *totalNumViewsCPU;
    AtomicU32 *totalNumInstancesCPU;
    AtomicU32 *totalNumLightsCPU;

    // This is used if on the CPU backend
    InstanceData *instancesCPU;
    PerspectiveCameraData *viewsCPU;
    LightDesc *lightsCPU;

    // World IDs (keys used in the key-value sorting)
    // Also only used when on the CPU backend
    uint64_t *instanceWorldIDsCPU;
    uint64_t *viewWorldIDsCPU;
    uint64_t *lightWorldIDsCPU;

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

    if (renderable.renderEntity == Entity::none()) {
        return;
    }

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
    if (renderable.renderEntity == Entity::none()) {
        return;
    }
    // Just update the instance data that is associated with this entity
#if defined(MADRONA_GPU_MODE)
    (void)e;

    InstanceData &data = ctx.get<InstanceData>(renderable.renderEntity);

    auto &system_state = ctx.singleton<RenderingSystemState>();
#else
    (void)renderable;

    // Just update the instance data that is associated with this entity
    auto &system_state = ctx.singleton<RenderingSystemState>();
    uint32_t instance_id = system_state.totalNumInstancesCPU->fetch_add<sync::acq_rel>(1);

    // Required for stable sorting on CPU
    system_state.instanceWorldIDsCPU[instance_id] = 
        ((uint64_t)ctx.worldID().idx << 32) | (uint64_t)e.id;

    InstanceData &data = system_state.instancesCPU[instance_id];
    data = ctx.get<InstanceData>(renderable.renderEntity);

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

uint32_t rgbToHex(Vector3 c) {
    float r = c.x;
    float g = c.y;
    float b = c.z;

    // Ensure the values are clamped between 0 and 1
    if (r < 0.0f) r = 0.0f;
    if (g < 0.0f) g = 0.0f;
    if (b < 0.0f) b = 0.0f;
    if (r > 1.0f) r = 1.0f;
    if (g > 1.0f) g = 1.0f;
    if (b > 1.0f) b = 1.0f;

    // Convert each component to an integer from 0 to 255
    uint8_t red = (uint8_t)(r * 255);
    uint8_t green = (uint8_t)(g * 255);
    uint8_t blue = (uint8_t)(b * 255);

    // Combine into a single uint32_t hex code
    return (red << 16) | (green << 8) | blue;
}

inline void lightUpdate(Context &ctx,
                        Entity e,
                        const Position &pos,
                        const LightDescDirection &dir,
                        const LightDescType &type,
                        const LightDescShadow &shadow,
                        const LightDescCutoffAngle &angle,
                        const LightDescIntensity &intensity,
                        const LightDescActive &active,
                        LightCarrier &carrier)
{
    if (carrier.light == Entity::none()) {
        return;
    }

    (void)e;

    LightDesc &desc = ctx.get<LightDesc>(carrier.light);

    desc.type = type.type;
    desc.castShadow = shadow.castShadow;
    desc.position = pos;
    desc.direction = dir;
    desc.cutoff = angle.cutoff;
    desc.intensity = intensity.intensity;
    desc.active = active.active;
}

inline void instanceTransformUpdateWithMat(Context &ctx,
                                           Entity e,
                                           const Position &pos,
                                           const Rotation &rot,
                                           const Scale &scale,
                                           const ObjectID &obj_id,
                                           const MaterialOverride &mat,
                                           const ColorOverride &color,
                                           const Renderable &renderable)
{
    if (renderable.renderEntity == Entity::none()) {
        return;
    }
    // Just update the instance data that is associated with this entity
#if defined(MADRONA_GPU_MODE)
    (void)e;

    InstanceData &data = ctx.get<InstanceData>(renderable.renderEntity);

    auto &system_state = ctx.singleton<RenderingSystemState>();
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

    data.matID = mat.matID;
    data.color = color.color;

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




#if 0
    uint32_t *morton_codes = state_mgr->getArchetypeComponent<
        RenderableArchetype, MortonCode>();
    
    WorldID *world_ids = state_mgr->getArchetypeComponent<
        RenderableArchetype, WorldID>();

    uint32_t current_world = 0;
    uint32_t current_world_offset = 0;

    for (int i = 0; 
         i < state_mgr->getArchetypeNumRows<RenderableArchetype>();
         ++i) {
        if (world_ids[i].idx != current_world) {
            current_world = world_ids[i].idx;
            current_world_offset = i;
        }

        uint32_t code = morton_codes[i];
        printf(USHORT_TO_BINARY_PATTERN " ", USHORT_TO_BINARY((code>>16)));
        printf(USHORT_TO_BINARY_PATTERN " \t", USHORT_TO_BINARY((code)));

        printf("(Leaf node %d)\t %d: (%d)\n", 
                i - current_world_offset, 
                world_ids[i].idx, 
                morton_codes[i]);
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
    registry.registerComponent<MaterialOverride>();
    registry.registerComponent<ColorOverride>();
    registry.registerComponent<LightDesc>();

    registry.registerComponent<LightDescDirection>();
    registry.registerComponent<LightDescType>();
    registry.registerComponent<LightDescShadow>();
    registry.registerComponent<LightDescCutoffAngle>();
    registry.registerComponent<LightDescIntensity>();
    registry.registerComponent<LightDescActive>();
    registry.registerComponent<LightCarrier>();

    registry.registerComponent<RGBOutputBuffer>(rgb_output_bytes);
    registry.registerComponent<DepthOutputBuffer>(depth_output_bytes);

    registry.registerComponent<RenderOutputIndex>();

    registry.registerComponent<RenderOutputRef>();
    registry.registerComponent<TLBVHNode>();

    registry.registerArchetype<RaycastOutputArchetype>();
    registry.registerArchetype<LightArchetype>();


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
                           Span<const TaskGraphNodeID> deps,
                           bool update_visual_properties)
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

    if (update_visual_properties) {
        instance_setup = builder.addToGraph<ParallelForNode<Context,
            instanceTransformUpdateWithMat,
                Entity,
                Position,
                Rotation,
                Scale,
                ObjectID,
                MaterialOverride,
                ColorOverride,
                Renderable
            >>({instance_setup});

        instance_setup = builder.addToGraph<ParallelForNode<Context,
             lightUpdate,
                Entity,
                Position,
                LightDescDirection,
                LightDescType,
                LightDescShadow,
                LightDescCutoffAngle,
                LightDescIntensity,
                LightDescActive,
                LightCarrier
            >>({instance_setup});
    }

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

    // Need to sort the instances, as well as the views

    // Need to sort by worlds first to handle deleted RenderableArchetypes
    auto sort_instances_by_world1 = 
        builder.addToGraph<CompactArchetypeNode<RenderableArchetype>>(
            {mortoncode_update});

    // Then sort by morton
    auto sort_instances_by_morton =
        builder.addToGraph<SortArchetypeNode<RenderableArchetype, MortonCode>>(
            {sort_instances_by_world1});

    auto post_instance_sort_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_instances_by_morton});

    // Then sort by world again to group up by world
    auto sort_instances_by_world2 = 
        builder.addToGraph<CompactArchetypeNode<RenderableArchetype>>(
            {post_instance_sort_reset_tmp});

#if 0
    auto sort_views = 
        builder.addToGraph<SortArchetypeNode<RenderCameraArchetype, WorldID>>(
            {post_instance_sort_reset_tmp});
#endif

    auto sort_views_world = builder.addToGraph<
        CompactArchetypeNode<RenderCameraArchetype>>(
            {sort_instances_by_world2});

    auto sort_views_index = builder.addToGraph<SortArchetypeNode<
        RenderCameraArchetype, RenderOutputIndex>>({sort_views_world});

    auto post_view_sort_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_views_index});

    auto sort_lights_world = builder.addToGraph<CompactArchetypeNode<
        LightArchetype>>({post_view_sort_reset_tmp});

#ifdef MADRONA_GPU_MODE
    auto export_counts = builder.addToGraph<ParallelForNode<Context,
        exportCountsGPU,
            RenderingSystemState
        >>({sort_lights_world});

    return export_counts;
#else
    return sort_lights_world;
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
        system_state.totalNumLights = bridge->totalNumLights;
#if !defined(MADRONA_GPU_MODE)
        // This is just an atomic counter (final value will be moved to
        // the totalNumViews/Instances variables).
        system_state.totalNumViewsCPU = bridge->totalNumViewsCPUInc;
        system_state.totalNumInstancesCPU = bridge->totalNumInstancesCPUInc;
        system_state.totalNumLightsCPU = bridge->totalNumLightsCPUInc;

        // This is only relevant for the CPU backend
        system_state.instancesCPU = bridge->instances;
        system_state.viewsCPU = bridge->views;
        system_state.lightsCPU = bridge->lights;
        system_state.instanceWorldIDsCPU = bridge->instancesWorldIDs;
        system_state.viewWorldIDsCPU = bridge->viewsWorldIDs;
        system_state.lightWorldIDsCPU = bridge->lightsWorldIDs;
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

    // Set default mat / color to not be overriden
    ctx.get<InstanceData>(render_entity).matID = -1;
    ctx.get<InstanceData>(render_entity).color = 0;
}

void disableEntityRenderable(Context &ctx,
                             Entity e)
{
    ctx.get<Renderable>(e).renderEntity = Entity::none();
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

void makeEntityLightCarrier(Context &ctx, Entity e)
{
    Entity light_e = ctx.makeEntity<LightArchetype>();
    ctx.get<LightCarrier>(e).light = light_e;

    ctx.get<LightDesc>(light_e) = LightDesc {
        .type = ctx.get<LightDescType>(e).type,
        .castShadow = ctx.get<LightDescShadow>(e).castShadow,
        .position = ctx.get<Position>(e),
        .direction = ctx.get<LightDescDirection>(e),
        .cutoff = ctx.get<LightDescCutoffAngle>(e).cutoff,
        .intensity = ctx.get<LightDescIntensity>(e).intensity,
        .active = ctx.get<LightDescActive>(e).active,
    };
}

#if 0
void configureLight(Context &ctx, Entity light, LightDesc desc)
{
    ctx.get<LightDesc>(light) = desc;
}
#endif

// Add this later when we decide to make the renderer more flexible
#if 0
void setEntityOutputIndex(Context &ctx, Entity e, uint32_t index)
{
    auto cam_entity = ctx.get<RenderCamera>(e).cameraEntity;
    ctx.get<RenderOutputIndex>(cam_entity).index = index;
}
#endif

}
