#include <madrona/optional.hpp>

namespace madrona::render {

struct RenderGraphBuilder::LogicalResource {
    enum class Type {
        Texture2D,
        Buffer,
    } type;

    struct Texture {
        Texture2DDesc desc;
        TextureHandle allocatedHandle;
    };

    struct Buffer {
        BufferDesc desc;
        BufferHandle allocatedHandle;
    };

    union {
        Texture tex2D;
        Buffer buffer;
    };

    bool writeHazard;

    LogicalResource *next;
};

struct RenderGraphBuilder::BarrierTaskArgs {
};

struct RenderGraphBuilder::TaskDesc {
    enum class Type {
        Raster,
        Compute,
        Copy,
        Barrier,
    } type;

    union {
        RasterTaskArgs raster;
        ComputeTaskArgs compute;
        CopyTaskArgs copy;
        BarrierTaskArgs barrier;
    };

    Optional<ParamBlock> paramBlock;

    void (*fn)(void *, GPU &, CommandBuffer cmd_buf);
    void *data;
    CountT numDataBytes;
    CountT dataAlignment;

    TaskDesc *next;
};

TaskResource RenderGraphBuilder::addTex2D(Texture2DDesc desc)
{
    auto *resource = addResource();
    resource->type = LogicalResource::Type::Texture2D;
    resource->tex2D = desc;

    return TaskResource { resource } ;
}

TaskResource RenderGraphBuilder::addBuffer(BufferDesc desc)
{
    auto *resource = addResource();
    resource->type = LogicalResource::Type::Buffer;
    resource->buffer = desc;

    return TaskResource { resource };
}

template <typename Fn>
void RenderGraphBuilder::addRasterTask(Fn &&fn, RasterTaskArgs args)
{
    TaskDesc *task = addTaskCommon(std::forward<Fn>(fn));

    task->type = TaskDesc::Type::Raster;
    task->raster = RasterTaskArgs {
        .vert = {
            .read = stashResourceHandles(args.vert.read),
        },
        .frag = {
            .read = stashResourceHandles(args.frag.read),
        },
        .attachments = {
            .readwrite = stashResourceHandles(args.attachments.readwrite),
            .write = stashResourceHandles(args.attachments.clear),
            .clear = stashResourceHandles(args.attachments.clear),
        },
    };
}

template <typename Fn>
void RenderGraphBuilder::addComputeTask(Fn &&fn, ComputeTaskArgs args)
{
    TaskDesc *task = addTaskCommon(std::forward<Fn>(fn));

    task->type = TaskDesc::Type::Compute;
    task->compute = ComputeTaskArgs {
        .read = stashResourceHandles(args.read),
        .write = stashResourceHandles(args.write),
        .readwrite = stashResourceHandles(args.readwrite),
        .forceAsync = args.forceAsync,
    };
}

template <typename Fn>
void RenderGraphBuilder::addCopyTask(Fn &&fn, CopyTaskArgs args)
{
    TaskDesc *task = addTaskCommon(std::forward<Fn>(fn));

    task->type = TaskDesc::Type::Copy;
    task->copy = CopyTaskArgs {
        .read = stashResourceHandles(args.read),
        .write = stashResourceHandles(args.write),
        .forceDMA = args.forceDMA,
    };
}

template <typename Fn>
RenderGraphBuilder::TaskDesc * RenderGraphBuilder::addTaskCommon(Fn &&fn)
{
    auto *fn_ptr = &RenderGraph::rasterTaskEntry<Fn>;

    auto *closure_store = alloc_->alloc<Fn>();
    *closure_store = fn;

    TaskDesc *new_task = alloc_->alloc<TaskDesc>();
    new_task->fn = fn_ptr;
    new_task->data = closure_store;
    new_task->numDataBytes = sizeof(Fn);
    new_task->dataAlignment = alignof(Fn);

    new_task->next = nullptr;
    task_list_tail_->next = new_task;
    task_list_tail_ = new_task;

    return new_task;
}

Span<TaskResource> RenderGraphBuilder::stashResourceHandles(
    Span<const TaskResource> resources)
{
    if (resources.size() == 0) {
        return { nullptr, 0 };
    }

    auto *dst = alloc_->allocN<TaskResource>(resources.size());
    utils::copyN<TaskResource>(dst, resources.data(), resources.size());

    return { dst, resources.size() };
}

RenderGraphBuilder::LogicalResource * RenderGraphBuilder::addResource()
{
    auto *resource = alloc_->alloc<LogicalResource>();
    resource->writeHazard = false;

    resource->next = nullptr;
    rsrc_list_tail_->next = resource;
    rsrc_list_tail_ = resource;

    return resource;
}

RenderGraphBuilder::LogicalResource * RenderGraphBuilder::getResource(
    TaskResource hdl)
{
    return (LogicalResource *)(hdl.hdl);
}

template <typename Fn>
void RenderGraph::rasterTaskEntry(void *data, GPU &gpu,
                                  RasterCommandList cmd_list)
{
    auto &closure = *(Fn *)data;
    closure(gpu, cmd_list);
}

template <typename Fn>
void RenderGraph::computeTaskEntry(void *data, GPU &gpu,
                                   ComputeCmdList cmd_list)
{
    auto &closure = *(Fn *)data;
    closure(gpu, cmd_list);
}

}
