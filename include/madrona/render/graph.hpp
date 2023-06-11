#pragma once

#include <madrona/render/gpu.hpp>
#include <madrona/span.hpp>
#include <madrona/stack_alloc.hpp>

namespace madrona::render {

enum class TaskType {
    Graphics,
    Compute,
    Copy,
};

struct TaskResource {
    void *hdl;
};

struct TaskArgs {
    Span<TaskResource> readonly = {};
    Span<TaskResource> readwrite = {};
    Span<TaskResource> writeonly = {};
    Span<TaskResource> fbAttachments = {};
};

class RenderGraphBuilder {
private:
    struct LogicalResource;
    struct TaskDesc;

public:
    struct TaskResource {
        LogicalResource *rsrc;
    };

    RenderGraphBuilder(StackAlloc &alloc);

    inline TaskResource addTex2D(Texture2DDesc desc);
    inline TaskResource addBuffer(CountT num_bytes);

    template <typename Fn>
    inline void addTask(Fn &&fn, TaskType type, TaskArgs args);

    RenderGraph build(GPU &gpu);

private:
    inline LogicalResource * addResource();
    inline LogicalResource * getResource(TaskResource hdl);

    StackAlloc *alloc_;
    StackAlloc::Frame alloc_start_;
    TaskDesc *task_list_head_;
    TaskDesc *task_list_tail_;
    LogicalResource *rsrc_list_head_;
    LogicalResource *rsrc_list_tail_;
    TaskDesc fake_task_;
    LogicalResource fake_rsrc_;
};

class RenderGraph {
public:
    ~RenderGraph();

private:
    struct Task;
    RenderGraph(void *data_buffer,
                Span<const Task> tasks,
                Span<TextureHandle> textures,
                Span<BufferHandle> buffers);

    template <typename Fn>
    static void taskEntry(void *data, GPU &gpu, CommandBuffer &cmd_buf);

    void *data_buffer_;
    Span<Task> tasks_;
    Span<TextureHandle> textures_;
    Span<BufferHandle> buffers_;

friend class RenderGraphBuilder;
};

}

#include "graph.inl"
