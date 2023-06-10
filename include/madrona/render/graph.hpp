#pragma once

#include <madrona/render/

namespace madrona::render {

class RenderGraph;

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

struct TaskArguments {
    Span<TaskResource> args;
};

class RenderGraphBuilder {
public:
    RenderGraphBuilder(StackAlloc &alloc);

    template <typename Fn>
    inline void addTask(Fn &&fn, TaskArguments args);

    RenderGraph build(GPUDevice &gpu);
private:
    struct TaskDesc {
        void (*fn)(void *data, GPU &);
        void *data;
        CountT numDataBytes;

        TaskDesc *next;
    };

    StackAlloc *alloc_;
};

class RenderGraph {
public:
        void (*fn)();
        void *data;
    };

private:
    template <typename Fn>
    static void taskEntry(void *data, GPU &);

    struct Task {
    };

friend class RenderGraphBuilder;
};

}

#include "graph.inl"
