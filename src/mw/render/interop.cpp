#include "interop.hpp"

#include <madrona/components.hpp>
#include <madrona/context.hpp>

namespace madrona::render {
using namespace base;
using namespace math;

inline void instanceTransformSetup(Context &ctx,
                                   const Position &pos,
                                   const Rotation &rot,
                                   const Scale &scale,
                                   const ObjectID &obj_id)
{
    RendererState &renderer_state = ctx.getSingleton<RendererState>();

#if defined(MADRONA_BATCHRENDER_RT)
    AtomicU32Ref count_atomic(
        renderer_state.numInstances->primitiveCount);

    uint32_t inst_idx = count_atomic.fetch_add<sync::relaxed>(1);

    AccelStructInstance &as_inst = renderer_state.tlasInstanceBuffer[inst_idx];

    Mat3x4 o2w = Mat3x4::fromTRS(pos + renderer_state.worldOffset, rot, scale);

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
#else
    AtomicU32Ref inst_count_atomic(*renderer_state.numInstances);
    uint32_t inst_idx = inst_count_atomic.fetch_add<sync::relaxed>(1);

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
                           const ViewSettings &view_settings)
{
    RendererState &renderer_state = ctx.getSingleton<RendererState>();
    int32_t view_idx = view_settings.viewID.idx;

#if defined(MADRONA_BATCHRENDER_RT)
    auto camera_pos =
        pos + view_settings.cameraOffset + renderer_state.worldOffset;

    PackedViewData &renderer_view = renderer_state.packedViews[view_idx];

    renderer_view.rotation = rot;
    renderer_view.posAndTanFOV.x = camera_pos.x;
    renderer_view.posAndTanFOV.y = camera_pos.y;
    renderer_view.posAndTanFOV.z = camera_pos.z;
    renderer_view.posAndTanFOV.w = view_settings.yScale;
#elif defined(MADRONA_BATCHRENDER_METAL)
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
                          RendererState &renderer_state)
{
    if (ctx.worldID().idx == 0) {
        *renderer_state.count_readback = renderer_state.numInstances->primitiveCount;
        renderer_state.numInstances->primitiveCount = 0;
    }
}

#endif

void RenderingSystem::registerTypes(ECSRegistry &registry)
{
    registry.registerComponent<ViewSettings>();
    registry.registerSingleton<RendererState>();
}

TaskGraph::NodeID RenderingSystem::setupTasks(TaskGraph::Builder &builder,
    Span<const TaskGraph::NodeID> deps)
{
    auto instance_setup = builder.addToGraph<ParallelForNode<Context,
        instanceTransformSetup,
        Position,
        Rotation,
        Scale,
        ObjectID>>(deps);

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
#endif
}

void RenderingSystem::reset([[maybe_unused]] Context &ctx)
{
#ifdef MADRONA_BATCHRENDER_METAL
    RendererState &renderer_state = ctx.getSingleton<RendererState>();
    *renderer_state.numViews = 0;
#endif
}

ViewSettings RenderingSystem::setupView([[maybe_unused]] Context &ctx,
                                        float vfov_degrees,
                                        float aspect_ratio,
                                        float z_near,
                                        math::Vector3 camera_offset,
                                        ViewID view_id)
{
    float fov_scale = 
#ifdef MADRONA_BATCHRENDER_METAL
        1.f / 
#endif
            tanf(helpers::toRadians(vfov_degrees * 0.5f));

#ifdef MADRONA_BATCHRENDER_METAL
    RendererState &renderer_state = ctx.getSingleton<RendererState>();
    (*renderer_state.numViews) += 1;
#endif

    float x_scale = fov_scale / aspect_ratio;
    float y_scale = fov_scale;

    return ViewSettings {
        x_scale,
        y_scale,
        z_near,
        camera_offset,
        view_id,
    };
}

void RendererState::init(Context &ctx, const RendererInit &renderer_init)
{
    RendererState &renderer_state = ctx.getSingleton<RendererState>();
    int32_t world_idx = ctx.worldID().idx;

#if defined(MADRONA_BATCHRENDER_RT)
    new (&renderer_state) RendererState {
        renderer_init.iface.tlasInstancesBase,
        renderer_init.iface.numInstances,
        renderer_init.iface.blases,
        renderer_init.iface.packedViews[world_idx],
        renderer_init.worldOffset,
#ifdef MADRONA_GPU_MODE
        renderer_init.iface.numInstancesReadback,
#endif
    };
#elif defined (MADRONA_BATCHRENDER_METAL)
    new (&renderer_state) RendererState {
        renderer_init.iface.views[world_idx],
        &renderer_init.iface.numViews[world_idx],
        renderer_init.iface.instanceData,
        renderer_init.iface.numInstances,
    };
#endif

}

}
