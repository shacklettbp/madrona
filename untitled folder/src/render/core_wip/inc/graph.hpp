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

struct TaskParamBlock {
};

struct TaskArgs {
    Span<const TaskResource> read = {};
    Span<const TaskResource> write = {};
    Span<const TaskResource> readwrite = {};
};

// Question: how we do we pass in the actual DescriptorSetLayout to the rendergraph
// Core issue: Ideally it would be great if the rendergraph could setup ParamBlock
// for each pass and bind physical resources to the paramblock. Then before the user code is called in the render loop it could bind the paramblock. and bind the pipeline before user's function is called. The problem is this makes a major assumption: 1. all the commands in the task need all the inputs and outputs specified. For RasterTasks, this is fairly straightforward for framebuffer attachments as a reasonable assumption seems to be that a given raster task MUST all be in a single render pass. However, what if the output of a prior pass is literally the index buffer for the next pass (or a acel structure build pass)? We don't even want to bind the pass inputs as a paramblock at that point
//
// - Also we don't want to do a 1-1 mapping between pipelines and graph tasks.
//
// A1: Does the rendergraph get the Shader object as input?
//  - No it doesn't. The rendergraph needs to be rebuilt on window resize due to
//    resource resizing. Compiling the shaders to pipelines needs to happen before
// A2: ParamBlock management is out of scope for the rendergraph. Pass a "setup" function to addXXX task when just receives
// resources as inputs and then does binding manually.
//   - Makes it hard / impossible to help manage descriptor set allocation across the pass. The issue is that if the setup functions are called sequentially, they need to create the concrete ParamBlocks at time of call. One option would be to pass the param block data requirements in the TaskArgs (I will need XYZ descriptor sets with N things => setup runs and these are passed in??). But wait, at this point why not just have a span of paramblocks in the task definition?
//   - Seems like it would allow a single addTask function. Could imagine helper functions layered on top like "addRenderPassTask" which does the renderpass setup for you.
//   - The most natural implementation of this seems like it would be the setup callback actually returning the run callback, which captures stuff in setup by value. Seems REALLLY ugly in terms of code you write though (nested lambdas).

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
    inline void addRasterTask(RasterTaskArgs args,
                              Fn &&fn);

    template <typename Fn>
    inline void addComputeTask(ComputeTaskArgs args,
                               Fn &&fn);

    template <typename Fn>
    inline void addCopyTask(CopyTaskArgs args, Fn &&fn);

    RenderGraph build(GPU &gpu, CountT num_frames);

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
