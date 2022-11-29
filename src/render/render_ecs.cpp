#include <madrona/render.hpp>
#include <madrona/components.hpp>
#include <madrona/context.hpp>

namespace madrona {
using namespace base;
using namespace math;

namespace render {

struct RendererState {
    std::atomic_uint32_t instanceCount;
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
    registry.registerComponent<ObjectID>();
    registry.registerSingleton<RendererState>();
}

inline void instanceAccelStructSetup(Context &ctx,
                                     const Position &pos,
                                     const Rotation &rot,
                                     const ObjectID &obj_id)
{
    RendererState &renderer_state = ctx.getSingleton<RendererState>();
    uint32_t inst_idx =
        renderer_state.instanceCount.fetch_add(1, std::memory_order_relaxed);

    uint64_t *blases = getBlases();

    AccelStructInstance &as_inst =
        getInstanceBuffer(ctx.worldID().idx)[inst_idx];

    Mat3x4 o2w = Mat3x4::fromTRS(pos, rot);
    
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


TaskGraph::NodeID RenderingSystem::setupTasks(TaskGraph::Builder &builder,
    Span<const TaskGraph::NodeID> deps)
{
    auto setup = builder.parallelForNode<Context,
                                         instanceAccelStructSetup,
                                         Position,
                                         Rotation,
                                         ObjectID>(deps);

    auto update_count = builder.parallelForNode<Context,
                                                updateInstanceCount,
                                                RendererState>({setup});

    return update_count;
}

}
}
