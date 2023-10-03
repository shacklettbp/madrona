#include "interop.hpp"
#include "madrona/render/batch_renderer_system.hpp"

#include <madrona/components.hpp>
#include <madrona/context.hpp>
#include <madrona/taskgraph.hpp>

#if defined(MADRONA_LINUX) or defined(MADRONA_WINDOWS) or defined(MADRONA_GPU_MODE)
#define MADRONA_BATCHRENDER_RT (1)
#elif defined(MADRONA_MACOS)
#define MADRONA_BATCHRENDER_METAL (1)
#endif

namespace madrona::render {
using namespace base;
using namespace math;

struct BatchRenderingSystemState {
    float aspectRatio;
    uint32_t maxViewsPerWorld;

    InstanceData *instances;
    uint32_t *instanceCounts;
};

void BatchRenderingSystem::registerTypes(ECSRegistry &registry,
                                         BatchRendererECSBridge *bridge)
{
    // These two need to be exported
    registry.registerComponent<InstanceData>();
    registry.registerComponent<PerspectiveCameraData>();

    registry.registerComponent<BatchRenderable>();
    registry.registerComponent<BatchRenderCameraEntity>();

    registry.registerArchetype<BatchRenderInstance>();

#if 0
    registry.registerArchetype<BatchRenderInstance>(
        ComponentSelector<InstanceData>(ComponentSelectImportFromVulkan));
#endif

    registry.registerArchetype<BatchRenderCameraInstance>();

    registry.registerSingleton<BatchRenderingSystemState>();

    // Figure out how to do this for the CPU backend.
#if defined MADRONA_GPU_MODE
    auto *state_mgr = mwGPU::getStateManager();
    bridge->instances = state_mgr->getArchetypeComponent<
        BatchRenderInstance,
        InstanceData>();
    bridge->views = state_mgr->getArchetypeComponent<
        BatchRenderCameraInstance,
        PerspectiveCameraData>();
    bridge->worldInstanceOffsets = state_mgr->getArchetypeSortOffsets<
        BatchRenderInstance>();
#else
    (void)bridge;
#endif
}

void BatchRenderingSystem::init(Context &ctx,
                                const BatchRendererECSBridge *bridge)
{
    auto &state = ctx.singleton<BatchRenderingSystemState>();

   state.aspectRatio = (float)bridge->renderWidth /
                       (float)bridge->renderHeight;
   state.maxViewsPerWorld = bridge->maxViewsPerWorld;
}

void BatchRenderingSystem::reset(Context &ctx)
{
    (void)ctx;
}

inline void instanceTransformUpdate(Context &ctx,
                                    const Position &pos,
                                    const Rotation &rot,
                                    const Scale &scale,
                                    const ObjectID &obj_id,
                                    const BatchRenderable &bu)
{
    // Just update the instance data
    InstanceData &data = ctx.get<InstanceData>(bu.renderEntity);
    data.position = pos;
    data.rotation = rot;
    data.scale = scale;
    data.worldID = ctx.worldID().idx;
    data.objectID = obj_id.idx;
}

inline void viewTransformUpdate(Context &ctx,
                                const Position &pos,
                                const Rotation &rot,
                                const BatchRenderCameraEntity &cam)
{
    Vector3 camera_pos = pos + cam.cameraOffset;

    PerspectiveCameraData &cam_data = 
        ctx.get<PerspectiveCameraData>(cam.cameraEntity);

    cam_data.position = Vector4::fromVector3(camera_pos, 1.0f);
    cam_data.rotation = rot.inv();
}

TaskGraphNodeID BatchRenderingSystem::setupTasks(TaskGraphBuilder &builder, 
                                                Span<const TaskGraphNodeID> deps)
{
    auto instance_update = builder.addToGraph<ParallelForNode<Context,
        instanceTransformUpdate,
            Position,
            Rotation,
            Scale,
            ObjectID,
            BatchRenderable
        >>(deps);

    auto view_update = builder.addToGraph<ParallelForNode<Context,
         viewTransformUpdate,
            Position,
            Rotation,
            BatchRenderCameraEntity
        >>({instance_update});

#ifdef MADRONA_GPU_MODE
    // Need to sort the instances, as well as the views
    auto sort_instances = 
        builder.addToGraph<SortArchetypeNode<BatchRenderInstance, WorldID>>(
            {view_update});

    auto post_instance_sort_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_instances});

    auto sort_views = 
        builder.addToGraph<SortArchetypeNode<BatchRenderCameraInstance, WorldID>>(
            {post_instance_sort_reset_tmp});

    auto post_view_sort_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_views});

    return post_view_sort_reset_tmp;

#endif

    return view_update;
}

void BatchRenderingSystem::makeEntityRenderable(Context &ctx,
                                                Entity e)
{
    Entity render_entity = ctx.makeEntity<BatchRenderInstance>();
    ctx.get<BatchRenderable>(e).renderEntity = render_entity;
}

void BatchRenderingSystem::attachEntityToView(Context &ctx,
                                              Entity e,
                                              float vfov_degrees,
                                              float z_near,
                                              int32_t view_idx,
                                              const math::Vector3 &camera_offset)
{
    Entity camera_entity = ctx.makeEntity<BatchRenderCameraInstance>();
    ctx.get<BatchRenderCameraEntity>(e) = { camera_entity, camera_offset };

    PerspectiveCameraData &cam_data = 
        ctx.get<PerspectiveCameraData>(camera_entity);

    auto &state = ctx.singleton<BatchRenderingSystemState>();

    float fov_scale = tanf(toRadians(vfov_degrees * 0.5f));
    float x_scale = fov_scale / state.aspectRatio;
    float y_scale = -fov_scale;

    cam_data = PerspectiveCameraData {
        { /* Position */ }, 
        { /* Rotation */ }, 
        x_scale, y_scale, 
        z_near, 
        view_idx,
        ctx.worldID().idx
    };
}

}
