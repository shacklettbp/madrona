#include <atomic>
#include <madrona/render.hpp>
#include <madrona/components.hpp>
#include <madrona/context.hpp>

namespace madrona {
using namespace base;
using namespace math;

namespace render {

struct RendererState {
    AccelStructInstance *tlasInstanceBuffer;
    uint32_t *instanceCountExport;
    uint64_t *blases;
    PackedViewData *packedViews;
    alignas(MADRONA_CACHE_LINE) std::atomic_uint32_t instanceCount;
};

void RenderingSystem::registerTypes(ECSRegistry &registry)
{
    registry.registerComponent<ViewSettings>();
    registry.registerSingleton<RendererState>();
}

inline void instanceAccelStructSetup(Context &ctx,
                                     const Position &pos,
                                     const Rotation &rot,
                                     const Scale &scale,
                                     const ObjectID &obj_id)
{
    RendererState &renderer_state = ctx.getSingleton<RendererState>();
    uint32_t inst_idx =
        renderer_state.instanceCount.fetch_add(1, std::memory_order_relaxed);

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
}

inline void updateRendererCounts(Context &,
                                 RendererState &renderer)
{
    uint32_t inst_count =
        renderer.instanceCount.load(std::memory_order_relaxed);
    renderer.instanceCount.store(0, std::memory_order_relaxed);

    *renderer.instanceCountExport = inst_count;
}

inline void updateViewData(Context &ctx,
                           const Position &pos,
                           const Rotation &rot,
                           const ViewSettings &view_settings)
{
    RendererState &renderer_state = ctx.getSingleton<RendererState>();

    int32_t view_idx = view_settings.viewID.idx;

    auto camera_pos = pos + view_settings.cameraOffset;

    PackedViewData &renderer_view = renderer_state.packedViews[view_idx];

    renderer_view.rotation = rot;
    renderer_view.posAndTanFOV.x = camera_pos.x;
    renderer_view.posAndTanFOV.y = camera_pos.y;
    renderer_view.posAndTanFOV.z = camera_pos.z;
    renderer_view.posAndTanFOV.w = view_settings.tanFOV;
}

TaskGraph::NodeID RenderingSystem::setupTasks(TaskGraph::Builder &builder,
    Span<const TaskGraph::NodeID> deps)
{
    auto instance_setup = builder.addToGraph<ParallelForNode<Context,
        instanceAccelStructSetup,
        Position,
        Rotation,
        Scale,
        ObjectID>>(deps);

    auto viewdata_update = builder.addToGraph<ParallelForNode<Context,
        updateViewData,
        Position,
        Rotation,
        ViewSettings>>({instance_setup});

    auto update_count = builder.addToGraph<ParallelForNode<Context,
        updateRendererCounts,
        RendererState>>({viewdata_update});

    return update_count;
}

void RenderingSystem::init(Context &ctx, const RendererInit &renderer_init)
{
    RendererState &renderer_state = ctx.getSingleton<RendererState>();
    new (&renderer_state) RendererState {
        renderer_init.iface.tlasInstancePtrs[ctx.worldID().idx],
        &renderer_init.iface.tlasInstanceCounts[ctx.worldID().idx],
        renderer_init.iface.blases,
        renderer_init.iface.packedViews[ctx.worldID().idx],
        0,
    };
}

ViewSettings RenderingSystem::setupView(Context &, float vfov_degrees,
                                        math::Vector3 camera_offset,
                                        ViewID view_id)
{
    float tan_fov = tanf(helpers::toRadians(vfov_degrees / 2.f));

    return ViewSettings {
        tan_fov,
        camera_offset,
        view_id,
    };
}

}
}
