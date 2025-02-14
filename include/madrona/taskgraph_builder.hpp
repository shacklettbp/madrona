#pragma once 

#include <madrona/fwd.hpp>
#include <madrona/taskgraph.hpp>
#include <madrona/context.hpp>

namespace madrona {

class TaskGraphManager;

// ID for a node in an under construction taskgraph. Used to link
// dependencies between nodes.
struct TaskGraphNodeID {
    uint32_t id;
};

// TaskGraphBuilder is constructed by the CPU or GPU backend
// and passed to the user's setupTasks function.
//
// After setupTasks returns, the backend calls TaskGraphBuilder::build
// to create the final taskgraph.
class TaskGraphBuilder {
public:
    // addToGraph is the primary function for end users.
    // Pass in the template'd type of the node to create and the
    // TaskGraphNodeIDs the node should depend on.
    // Returns a TaskGraphNodeID for the newly added taskgraph node.
    // Example:
    //   auto next_node = builder.addToGraph<ParallelForNode<
    //      MyContext, mySystem, MyComponent>>({other_node});
    template <typename NodeT>
    inline TaskGraphNodeID addToGraph(
        Span<const TaskGraphNodeID> dependencies);

    // The below functions in this class are only necessary when implementing
    // custom taskgraph nodes and should not be used by the majority of users.
    // Skip to the builtin taskgraph nodes below (eg ParallelForNode).

    struct DataID {
        int32_t id;
    };

    template <typename NodeT>
    struct TypedDataID : DataID {};

    template <typename NodeT, typename... Args>
    TypedDataID<NodeT> constructNodeData(Args &&...args);

    template <auto fn, typename NodeT>
    TaskGraphNodeID addNodeFn(TypedDataID<NodeT> data,
                              Span<const TaskGraphNodeID> dependencies,
                              Optional<TaskGraphNodeID> parent_node =
                                  Optional<TaskGraphNodeID>::none());

    template <typename NodeT, typename... Args>
    TaskGraphNodeID addDefaultNode(Span<const TaskGraphNodeID> dependencies,
                                   Args && ...args);

    template <typename NodeT>
    NodeT & getDataRef(TypedDataID<NodeT> data_id);

private:
    TaskGraphBuilder(uint32_t taskgraph_id, const WorkerInit &init);

    // Called by the backend to build the taskgraph.
    TaskGraph build();

    TaskGraphNodeID registerNode(uint32_t data_idx,
        void (*fn)(NodeBase *, Context *, TaskGraph *),
        Span<const TaskGraphNodeID> dependencies,
        Optional<TaskGraphNodeID> parent_node);

    struct StagedNode {
        TaskGraph::Node node;
        int32_t parentID;
        uint32_t dependencyOffset;
        uint32_t numDependencies;
    };

    StateManager *state_mgr_;
    StateCache *state_cache_;
#ifdef MADRONA_MW_MODE
    uint32_t world_id_;
#endif
    uint32_t taskgraph_id_;
    DynArray<StagedNode> staged_;
    DynArray<TaskGraph::NodeData> node_datas_;
    DynArray<TaskGraphNodeID> all_dependencies_;

friend class TaskGraphManager;
};

class TaskGraphManager {
public:
    TaskGraphManager(CountT num_taskgraphs, const WorkerInit &init);
    ~TaskGraphManager();

    // Create a new TaskgraphBuilder for building a task graph
    template <EnumType EnumT>
    TaskGraphBuilder & init(EnumT taskgraph_id);
    TaskGraphBuilder & init(uint32_t taskgraph_id);

    // Called by the backend after user's setupTasks populates the graphs
    HeapArray<TaskGraph> constructGraphs();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Builtin taskgraph nodes

// ParallelForNode is the core of the ECS taskgraph. This node will
// call Fn in parallel over every entity matching the list of Component types
// passed in the signature.
// For example, given:
//     void mySystem(MyContext &ctx,
//                   Position &position,
//                   Rotation &rotation);
//
// The following node type iterates over each entity with Position & Rotation:
//     ParallelForNode<MyContext, mySystem, Position, Rotation>
//
// Make sure not to mark your function (in this case mySystem) as static.
// This currently causes problems in the GPU backend.
//
// TODO: Make sure to either fix this, or throw an error in the case of this
// happening.
template <typename ContextT, auto Fn, typename ...ComponentTs>
class ParallelForNode : public NodeBase {
public:
    ParallelForNode(Query<ComponentTs...> &&query);

    inline void run(Context &ctx_base, TaskGraph &taskgraph);

    static TaskGraphNodeID addToGraph(
        StateManager &state_mgr,
        TaskGraphBuilder &builder,
        Span<const TaskGraphNodeID> dependencies);

private:
    Query<ComponentTs...> query_;
};

// This node resets the temporary bump allocator accessible through
// Context::tmpAlloc
class ResetTmpAllocNode : public NodeBase {
public:
    inline void run(Context &ctx, TaskGraph &);

    static TaskGraphNodeID addToGraph(
        StateManager &,
        TaskGraphBuilder &builder,
        Span<const TaskGraphNodeID> dependencies);
};

// This node destroys all the temporary entities of archetype ArchetypeT
template <typename ArchetypeT>
class ClearTmpNode : public NodeBase {
public:
    inline void run(Context &ctx, TaskGraph &taskgraph);

    static TaskGraphNodeID addToGraph(
        StateManager &,
        TaskGraphBuilder &builder,
        Span<const TaskGraphNodeID> dependencies);
};

class SortArchetypeNodeBase : public NodeBase {
public:
    SortArchetypeNodeBase(uint32_t archetype_id,
                          uint32_t component_id);

    void run(Context &ctx, TaskGraph &taskgraph);

    static TaskGraphNodeID addToGraph(
        StateManager &state_mgr,
        TaskGraphBuilder &builder,
        Span<const TaskGraphNodeID> dependencies,
        uint32_t archetype_id,
        uint32_t component_id);

private:
    uint32_t archetype_id_;
    uint32_t component_id_;
};

template <typename ArchetypeT, typename ComponentT>
class SortArchetypeNode : public SortArchetypeNodeBase {
public:
    static TaskGraphNodeID addToGraph(
        StateManager &state_mgr,
        TaskGraphBuilder &builder,
        Span<const TaskGraphNodeID> dependencies);
};

class CompactArchetypeNodeBase : public NodeBase {
public:
    CompactArchetypeNodeBase(uint32_t archetype_id);

    void run(Context &ctx, TaskGraph &taskgraph);

    static TaskGraphNodeID addToGraph(
        StateManager &state_mgr,
        TaskGraphBuilder &builder,
        Span<const TaskGraphNodeID> dependencies,
        uint32_t archetype_id);

private:
    uint32_t archetype_id_;
};

template <typename ArchetypeT>
class CompactArchetypeNode : public CompactArchetypeNodeBase {
public:
    static TaskGraphNodeID addToGraph(
        StateManager &state_mgr,
        TaskGraphBuilder &builder,
        Span<const TaskGraphNodeID> dependencies);
};

}

#include "taskgraph_builder.inl"
