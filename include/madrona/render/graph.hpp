#pragma once

#include <madrona/render/gpu.hpp>
#include <madrona/span.hpp>
#include <madrona/stack_alloc.hpp>

namespace madrona::render {

enum class TaskType {
    Compute,
    Raster,
};

struct TaskResource {
    enum class Type {
        Buffer,
        Texture2D,
    } type;

    union {
        Texture2DDesc tex2D;
        BufferDesc buffer;
    };
};

class RenderGraphBuilder {
public:
    RenderGraphBuilder(StackAlloc &alloc);

    template <typename Fn>
    inline void addTask(Fn &&fn,
        TaskType type,
        Span<TaskResource> read_resources,
        Span<TaskResource> write_resources);

    RenderGraph build(GPU &gpu);
private:
    struct TaskDesc {
        void (*fn)(void *data, GPU &);
        void *data;
        CountT numDataBytes;

        TaskType type;

        TaskResource *readResources;
        TaskResource *writeResources;

        TaskDesc *next;
    };
    StackAlloc *alloc_;
    StackAlloc::Frame alloc_start_;
    TaskDesc *task_list_head_;
    TaskDesc *task_list_tail_;
    TaskDesc empty_;

};

class RenderGraph {
public:

private:
    template <typename Fn>
    static void taskEntry(void *data, GPU &);

    struct Task {
    };

friend class RenderGraphBuilder;
};

}

#include "graph.inl"
