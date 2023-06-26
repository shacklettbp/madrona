#pragma once

#include <madrona/dyn_array.hpp>
#include <madrona/heap_array.hpp>
#include <madrona/span.hpp>
#include <madrona/state.hpp>
#include <madrona/context.hpp>

#include <functional>
#include <thread>

namespace madrona {
struct NodeBase {};

class TaskGraph {
private:
    static inline constexpr uint32_t maxNodeDataBytes = 128;
    static inline constexpr uint32_t maxNodeDataAlign = 64;
    struct alignas(maxNodeDataAlign) NodeData {
        char userData[maxNodeDataBytes];
    };
    static_assert(sizeof(NodeData) == 128);

    struct Node {
        void (*fn)(NodeBase *, Context *);
        uint32_t dataIDX;
        uint32_t numChildren;
    };

public:
    struct NodeID {
        uint32_t id;
    };

    struct DataID {
        int32_t id;
    };

    template <typename NodeT>
    struct TypedDataID : DataID {};

    class Builder {
    public:
        Builder(Context &ctx);

        template <typename NodeT, typename... Args>
        TypedDataID<NodeT> constructNodeData(Args &&...args);

        template <auto fn, typename NodeT>
        NodeID addNodeFn(TypedDataID<NodeT> data,
                         Span<const NodeID> dependencies,
                         Optional<NodeID> parent_node =
                             Optional<NodeID>::none());

        template <typename NodeT, typename... Args>
        TaskGraph::NodeID addDefaultNode(Span<const NodeID> dependencies,
                                         Args && ...args);

        template <typename NodeT>
        inline NodeID addToGraph(Span<const NodeID> dependencies);

        template <typename NodeT>
        NodeT & getDataRef(TypedDataID<NodeT> data_id);

        TaskGraph build();

    private:
        NodeID registerNode(uint32_t data_idx,
                            void (*fn)(NodeBase *, Context *),
                            Span<const NodeID> dependencies,
                            Optional<NodeID> parent_node);

        struct StagedNode {
            Node node;
            int32_t parentID;
            uint32_t dependencyOffset;
            uint32_t numDependencies;
        };

        Context *ctx_;
        DynArray<StagedNode> staged_;
        DynArray<NodeData> node_datas_;
        DynArray<NodeID> all_dependencies_;
    };

    TaskGraph(const TaskGraph &) = delete;

    void run(Context *ctx);

private:
    TaskGraph(HeapArray<Node> &&sorted_nodes,
              HeapArray<NodeData> &&node_datas);

    HeapArray<Node> sorted_nodes_;
    HeapArray<NodeData> node_datas_;

friend class Builder;
};

template <typename ContextT, auto Fn, typename ...ComponentTs>
class ParallelForNode : public NodeBase {
public:
    ParallelForNode(Context &ctx);

    inline void run(Context *ctx_base);

    static TaskGraph::NodeID addToGraph(
            Context &ctx,
            TaskGraph::Builder &builder,
            Span<const TaskGraph::NodeID> dependencies);

private:
    Query<ComponentTs...> query_;
};

class ResetTmpAllocNode : public NodeBase {
public:
    inline void run(Context *ctx);

    static TaskGraph::NodeID addToGraph(
            Context &ctx,
            TaskGraph::Builder &builder,
            Span<const TaskGraph::NodeID> dependencies);
};

template <typename ArchetypeT>
class ClearTmpNode : public NodeBase {
public:
    inline void run(Context *ctx);

    static TaskGraph::NodeID addToGraph(
            Context &ctx,
            TaskGraph::Builder &builder,
            Span<const TaskGraph::NodeID> dependencies);
};

}

#include "taskgraph.inl"
