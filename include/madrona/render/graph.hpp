#pragma once

namespace madrona::render {

class RenderGraph;

struct RenderTask {
};

class RenderGraphBuilder {
public:
    RenderGraphBuilder();

    template <typename Fn>
    inline void addTask(Fn &&fn);

    RenderGraph build(GPUDevice &gpu);
private:
    DynArray<TaskDefn> task_defns_;
};

class RenderGraph {
public:
    struct TaskDefn {
        void (*fn)();
        void *data;
    };


private:
    struct Task {
    };

friend class RenderGraphBuilder;
};

}

#include "graph.inl"
