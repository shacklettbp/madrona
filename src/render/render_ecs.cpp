#include <atomic>
#include <madrona/render.hpp>
#include <madrona/components.hpp>
#include <madrona/context.hpp>

namespace madrona {
using namespace base;
using namespace math;

namespace render {

// FIXME this is a copy of the PackedCamera / ViewData
// struct in render/vk/shaders/shader_common.h
struct ViewData {
    Quat rotation;
    Vector4 posAndTanFOV;
};

struct RendererState {
    std::atomic_uint32_t instanceCount;

    static inline std::atomic_int32_t viewOffset = 0;
};

static inline AccelStructInstance * getInstanceBuffer(int32_t world_idx)
{
    return (AccelStructInstance *)
        (mwGPU::GPUImplConsts::get().rendererASInstancesAddrs[world_idx]);
}

static inline void exportInstanceCount(int32_t world_idx,
                                       uint32_t instance_count)
{
    ((uint32_t *)mwGPU::GPUImplConsts::get().
        rendererInstanceCountsAddr)[world_idx] = instance_count;
}

static inline uint64_t * getBlases()
{
    return (uint64_t *)mwGPU::GPUImplConsts::get().rendererBLASesAddr;
}

void RenderingSystem::registerTypes(ECSRegistry &registry)
{
    registry.registerComponent<ActiveView>();
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

    uint64_t *blases = getBlases();

    AccelStructInstance &as_inst =
        getInstanceBuffer(ctx.worldID().idx)[inst_idx];

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
    as_inst.accelerationStructureReference = blases[obj_id.idx];
}

inline void updateInstanceCount(Context &ctx,
                                RendererState &renderer)
{
    uint32_t inst_count =
        renderer.instanceCount.load(std::memory_order_relaxed);
    renderer.instanceCount.store(0, std::memory_order_relaxed);

    exportInstanceCount(ctx.worldID().idx, inst_count);
}

inline void updateViewData(Context &,
                           const Position &pos,
                           const Rotation &rot,
                           const ActiveView &view)
{
    auto viewdatas =
        (ViewData *)mwGPU::GPUImplConsts::get().rendererViewDatasAddr;

    auto camera_pos = pos + view.cameraOffset;

    ViewData &renderer_view = viewdatas[view.viewIdx];
    renderer_view.rotation = rot;
    renderer_view.posAndTanFOV.x = camera_pos.x;
    renderer_view.posAndTanFOV.y = camera_pos.y;
    renderer_view.posAndTanFOV.z = camera_pos.z;
    renderer_view.posAndTanFOV.w = view.tanFOV;
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
        ActiveView>({instance_setup});

    auto update_count = builder.addToGraph<ParallelForNode<Context,
        updateInstanceCount,
        RendererState>({viewdata_update});

    return update_count;
}

void RenderingSystem::init(Context &ctx)
{
    RendererState &renderer_state = ctx.getSingleton<RendererState>();
    new (&renderer_state) RendererState {
        0,
    };
}

ActiveView RenderingSystem::setupView(Context &, float vfov_degrees,
                                      math::Vector3 camera_offset)
{
    int32_t view_offset = RendererState::viewOffset.fetch_add(1,
        std::memory_order_relaxed);

    float tan_fov = tanf(helpers::toRadians(vfov_degrees / 2.f));

    return ActiveView {
        tan_fov, 
        camera_offset,
        view_offset,
    };
}

}
}
