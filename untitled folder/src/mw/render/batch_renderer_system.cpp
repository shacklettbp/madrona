#include "interop.hpp"

#include <madrona/components.hpp>
#include <madrona/context.hpp>

#if defined(MADRONA_LINUX) or defined(MADRONA_WINDOWS) or defined(MADRONA_GPU_MODE)
#define MADRONA_BATCHRENDER_RT (1)
#elif defined(MADRONA_MACOS)
#define MADRONA_BATCHRENDER_METAL (1)
#endif

namespace madrona::render {
using namespace base;
using namespace math;

inline void clearInstanceCount(Context &,
                               const BatchRendererState &renderer_state)
{
#ifdef MADRONA_BATCHRENDER_METAL
    *(renderer_state.numInstances) = 0;
#endif
}

inline void instanceTransformSetup(Context &ctx,
                                   const Position &pos,
                                   const Rotation &rot,
                                   const Scale &scale,
                                   const ObjectID &obj_id)
{
    BatchRendererState &renderer_state = ctx.singleton<BatchRendererState>();

#if defined(MADRONA_BATCHRENDER_RT)
    AtomicU32Ref count_atomic(
        renderer_state.numInstances->primitiveCount);

    uint32_t inst_idx = count_atomic.fetch_add_relaxed(1);

    AccelStructInstance &as_inst = renderer_state.tlasInstanceBuffer[inst_idx];

    Mat3x4 o2w = Mat3x4::fromTRS(pos, rot, scale);

    as_inst.transform.matrix[0][0] = o2w.cols[0].x;
    as_inst.transform.matrix[0][1] = o2w.cols[1].x;
    as_inst.transform.matrix[0][2] = o2w.cols[2].x;
    as_inst.transform.matrix[0][3] = o2w.cols[3].x;

    as_inst.transform.matrix[1][0] = o2w.cols[0].y;
    as_inst.transform.matrix[1][1] = o2w.cols[1].y;
    as_inst.transform.matrix[1][2] = o2w.cols[2].y;
    as_inst.transform.matrix[1][3] = o2w.cols[3].y;

    as_inst.transform.matrix[2][0] = o2w.cols[0].z;
    as_inst.transform.matrix[2][1] = o2w.cols[1].z;
    as_inst.transform.matrix[2][2] = o2w.cols[2].z;
    as_inst.transform.matrix[2][3] = o2w.cols[3].z;

    as_inst.instanceCustomIndex = obj_id.idx;
    as_inst.mask = 1;
    as_inst.instanceShaderBindingTableRecordOffset = 0;
    as_inst.flags = 0;
    as_inst.accelerationStructureReference = renderer_state.blases[obj_id.idx];
#elif defined(MADRONA_BATCHRENDER_METAL)
    AtomicU32Ref inst_count_atomic(*renderer_state.numInstances);
    uint32_t inst_idx = inst_count_atomic.fetch_add_relaxed(1);

    renderer_state.instanceData[inst_idx] = InstanceData {
        pos,
        rot,
        scale,
        obj_id.idx,
        ctx.worldID().idx,
    };
#endif
}

inline void updateViewData(Context &ctx,
                           const Position &pos,
                           const Rotation &rot,
                           const BatchRenderCamera &view_settings)
{
    BatchRendererState &renderer_state = ctx.singleton<BatchRendererState>();
    int32_t view_idx = view_settings.viewIDX;

#if defined(MADRONA_BATCHRENDER_RT)
    auto camera_pos =
        pos + view_settings.cameraOffset;

    PackedViewData &renderer_view = renderer_state.packedViews[view_idx];

    renderer_view.rotation = rot;
    renderer_view.posAndTanFOV.x = camera_pos.x;
    renderer_view.posAndTanFOV.y = camera_pos.y;
    renderer_view.posAndTanFOV.z = camera_pos.z;
    renderer_view.posAndTanFOV.w = view_settings.yScale;
#else
    Vector3 camera_pos = pos + view_settings.cameraOffset;

    renderer_state.views[view_idx] = PerspectiveCameraData {
        camera_pos,
        rot.inv(),
        view_settings.xScale,
        view_settings.yScale,
        view_settings.zNear,
        {},
    };
#endif
}

#ifdef MADRONA_GPU_MODE

inline void readbackCount(Context &ctx,
                          BatchRendererState &renderer_state)
{
    if (ctx.worldID().idx == 0) {
        *renderer_state.count_readback = renderer_state.numInstances->primitiveCount;
        renderer_state.numInstances->primitiveCount = 0;
    }
}

#endif

void BatchRenderingSystem::registerTypes(ECSRegistry &registry)
{
    registry.registerComponent<BatchRenderCamera>();
    registry.registerSingleton<BatchRendererState>();
}

TaskGraphNodeID BatchRenderingSystem::setupTasks(TaskGraphBuilder &builder,
    Span<const TaskGraphNodeID> deps)
{
    // FIXME: It feels like we should have persistent slots for renderer
    // state rather than needing to continually reset the instance count
    // and recreate the buffer. However, this might be hard to handle with
    // double buffering
    auto instance_clear = builder.addToGraph<ParallelForNode<Context,
        clearInstanceCount,
        BatchRendererState>>(deps);

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
        BatchRenderCamera>>({instance_setup});

#ifdef MADRONA_GPU_MODE
    auto readback_count = builder.addToGraph<ParallelForNode<Context,
        readbackCount,
        BatchRendererState>>({viewdata_update});

    return readback_count;
#else
    return viewdata_update;
#endif
}

void BatchRenderingSystem::reset([[maybe_unused]] Context &ctx)
{
#if defined(MADRONA_BATCHRENDER_METAL)
    BatchRendererState &renderer_state = ctx.singleton<BatchRendererState>();
    *renderer_state.numViews = 0;
#endif
}

BatchRenderCamera BatchRenderingSystem::setupView(Context &ctx,
                                        float vfov_degrees,
                                        float z_near,
                                        math::Vector3 camera_offset,
                                        int32_t view_idx)
{
    BatchRendererState &renderer_state = ctx.singleton<BatchRendererState>();

    float fov_scale =
#ifndef MADRONA_BATCHRENDER_RT
        1.f / 
#endif
            tanf(toRadians(vfov_degrees * 0.5f));

#if defined(MADRONA_BATCHRENDER_METAL)
    (*renderer_state.numViews) += 1;
#endif

    float x_scale = fov_scale / renderer_state.aspectRatio;
    float y_scale =
#ifndef MADRONA_BATCHRENDER_METAL
        -
#endif
        fov_scale;

    return BatchRenderCamera {
        x_scale,
        y_scale,
        z_near,
        camera_offset,
        view_idx,
    };
}

void BatchRenderingSystem::init(
    Context &ctx, const BatchRendererECSBridge *bridge)
{
    BatchRendererState &renderer_state = ctx.singleton<BatchRendererState>();
    int32_t world_idx = ctx.worldID().idx;

    new (&renderer_state) BatchRendererState {
#if defined(MADRONA_BATCHRENDER_RT)
        bridge->tlasInstancesBase,
        bridge->numInstances,
        bridge->blases,
        bridge->packedViews[world_idx],
#ifdef MADRONA_GPU_MODE
        bridge->numInstancesReadback,
#endif
#elif defined (MADRONA_BATCHRENDER_METAL)
        bridge->views[world_idx],
        &bridge->numViews[world_idx],
        bridge->instanceData,
        bridge->numInstances,
#endif
        bridge->renderWidth,
        bridge->renderHeight,
        float(bridge->renderWidth) /
            float(bridge->renderHeight),
    };
}

}
