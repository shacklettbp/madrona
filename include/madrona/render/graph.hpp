#pragma once

#include <madrona/render/gpu.hpp>
#include <madrona/span.hpp>
#include <madrona/stack_alloc.hpp>

namespace madrona::render {

struct TaskResource {
    void *hdl;
};

struct RasterTaskArgs {
    struct Vertex {
        Span<const TaskResource> read;
    } vert = {{}};

    struct Fragment {
        Span<const TaskResource> read;
    } frag = {{}};

    struct Attachments {
        Span<const TaskResource> readwrite;
        Span<const TaskResource> write;
        Span<const TaskResource> clear;
    } attachments = {{}, {}, {}};
};

struct ComputeTaskArgs {
    Span<const TaskResource> read = {};
    Span<const TaskResource> write = {};
    Span<const TaskResource> readwrite = {};
    bool forceAsync = false;
};

struct CopyTaskArgs {
    Span<const TaskResource> read = {};
    Span<const TaskResource> write = {};
    bool forceDMA = false;
};

class RenderGraphBuilder {
private:
    struct LogicalResource;
    struct BarrierTaskArgs;
    struct TaskDesc;

public:
    RenderGraphBuilder(StackAlloc &alloc);

    inline TaskResource addTex2D(Texture2DDesc desc);
    inline TaskResource addBuffer(BufferDesc desc);

    template <typename Fn>
    inline void addRasterTask(Fn &&fn, RasterTaskArgs args);

    template <typename Fn>
    inline void addComputeTask(Fn &&fn, ComputeTaskArgs args);

    template <typename Fn>
    inline void addCopyTask(Fn &&fn, CopyTaskArgs args);

    RenderGraph build(GPU &gpu, CountT num_inflight);

private:
    inline LogicalResource * addResource();
    inline LogicalResource * getResource(TaskResource hdl);

    template <typename Fn>
    inline TaskDesc * addTaskCommon(Fn &&fn);
    inline Span<TaskResource> stashResourceHandles(
        Span<const TaskResource> resources);

    inline bool checkHazards(TaskDesc *task);
    inline bool checkHazards(Span<const TaskResource> resources);
    inline void markHazards(TaskDesc *task);
    inline void markHazards(Span<const TaskResource> resources);
    inline TaskDesc * addBarrier(TaskDesc *prev, TaskDesc *next);

    StackAlloc *alloc_;
    StackAlloc::Frame alloc_start_;
    TaskDesc *task_list_head_;
    TaskDesc *task_list_tail_;
    LogicalResource *rsrc_list_head_;
    LogicalResource *rsrc_list_tail_;
};

class RenderGraph {
public:
    ~RenderGraph();

    void submit(GPU &gpu);

private:
    struct Task;
    RenderGraph(void *data_buffer,
                Span<const Task> tasks,
                Span<const TextureHandle> textures,
                Span<const BufferHandle> buffers);

    template <typename Fn>
    static void rasterTaskEntry(void *data, GPU &gpu,
                                RasterCmdList cmd_list);

    template <typename Fn>
    static void computeTaskEntry(void *data, GPU &gpu,
                                 ComputeCmdList cmd_list);

    template <typename Fn>
    static void copyTaskEntry(void *data, GPU &gpu,
                              CopyCmdList cmd_list);

    void *data_buffer_;
    Span<const Task> tasks_;
    Span<const TextureHandle> textures_;
    Span<const BufferHandle> buffers_;

friend class RenderGraphBuilder;
};

}

#include "graph.inl"
