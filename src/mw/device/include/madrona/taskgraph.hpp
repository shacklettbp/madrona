#pragma once

#include <madrona/span.hpp>
#include <madrona/query.hpp>
#include <madrona/state.hpp>
#include <madrona/inline_array.hpp>

#include "mw_gpu/const.hpp"
#include "mw_gpu/worker_init.hpp"
#include "mw_gpu/megakernel_consts.hpp"

#include <cuda/barrier>
#include <cuda/std/tuple>

//#define LIMIT_ACTIVE_THREADS
// #define LIMIT_ACTIVE_BLOCKS
// #define FETCH_MULTI_INVOCATIONS

namespace madrona {

class TaskGraph;
class TaskGraphManager;

namespace mwGPU {
    inline TaskGraph & getTaskGraph(uint32_t taskgraph_idx);
}

struct NodeBase {
    // Number of threads to dispatch for the next node in the graph of the same 
    // node type (e.g. SortArchetypeNode).
    uint32_t numDynamicInvocations;
};

class TaskGraph {
private:
    static inline constexpr uint32_t maxNodeDataBytes = 256;
    struct alignas(maxNodeDataBytes) NodeData {
        char userData[maxNodeDataBytes];
    };
    static_assert(sizeof(NodeData) == 256);

    struct Node {
        uint32_t dataIDX;
        uint32_t fixedCount;
        uint32_t funcID;
        uint32_t numChildren;
        uint32_t numThreadsPerInvocation;

        AtomicU32 curOffset;
        AtomicU32 numRemaining;
        AtomicU32 totalNumInvocations;
    };

public:
    struct NodeID {
        int32_t id;
    };

    struct DataID {
        int32_t id;
    };

    template <typename NodeT>
    struct TypedDataID : DataID {};

    class Builder {
    public:
        Builder(uint32_t taskgraph_id,
                int32_t max_nodes,
                int32_t max_node_datas,
                int32_t max_num_dependencies);
        ~Builder();

        template <typename NodeT, typename... Args>
        TypedDataID<NodeT> constructNodeData(Args &&...args);

        template <auto fn, typename NodeT>
        NodeID addNodeFn(TypedDataID<NodeT> data,
                         Span<const NodeID> dependencies,
                         Optional<NodeID> parent_node =
                             Optional<NodeID>::none(),
                         uint32_t fixed_num_invocations = 0,
                         uint32_t num_threads_per_invocation = 1);

        template <typename NodeT, int32_t count = 1, typename... Args>
        NodeID addOneOffNode(Span<const NodeID> dependencies,
                             Args && ...args);

        template <typename NodeT, typename... Args>
        NodeID addDynamicCountNode(Span<const NodeID> dependencies,
                                   uint32_t num_threads_per_invocation,
                                   Args && ...args);

        template <typename NodeT>
        inline NodeID addToGraph(Span<const NodeID> dependencies);

        template <typename NodeT>
        NodeT & getDataRef(TypedDataID<NodeT> data_id);

        inline uint32_t getTaskgraphID() const;

        void build(TaskGraph *out);

    private:
        template <typename NodeT>
        static void dynamicCountWrapper(NodeT *node, int32_t);

        NodeID registerNode(uint32_t data_idx,
                            uint32_t fixed_count,
                            uint32_t num_threads_per_invocation,
                            uint32_t func_id,
                            Span<const NodeID> dependencies,
                            Optional<NodeID> parent_node);

        struct StagedNode {
            Node node;
            int32_t parentID;
            uint32_t dependencyOffset;
            uint32_t numDependencies;
        };

        StagedNode *staged_;
        int32_t num_nodes_;
        NodeData *node_datas_;
        int32_t num_datas_;
        NodeID *all_dependencies_;
        uint32_t taskgraph_id_;
        uint32_t num_dependencies_;
        uint32_t max_num_nodes_;
        uint32_t max_num_node_datas_;
        uint32_t max_num_dependencies_;
    };

    enum class WorkerState {
        Run,
        PartialRun,
        Loop,
        Exit,
    };

    TaskGraph(const TaskGraph &) = delete;

    ~TaskGraph();

    void init(int32_t start_node_idx, int32_t end_node_idx,
              int32_t num_blocks_per_sm);

    WorkerState getWork(NodeBase **node_data,
                        uint32_t *run_func_id,
                        uint32_t *run_node_id,
                        int32_t *run_offset);

    void finishWork(bool lane_executed);

    static inline WorldBase * getWorld(int32_t world_idx);

    template <typename ContextT>
    static ContextT makeContext(WorldID world_id);

    template <typename NodeT>
    NodeT & getNodeData(TypedDataID<NodeT> data_id);

    struct BlockState;
private:
    template <typename ContextT, bool> struct WorldTypeExtract;

    TaskGraph(Node *nodes, uint32_t num_nodes, NodeData *node_datas);

    inline void updateBlockState();
    inline uint32_t computeNumInvocations(Node &node);

    Node *sorted_nodes_;
    uint32_t num_nodes_;
    uint32_t end_node_idx_;
    NodeData *node_datas_;
    AtomicU32 cur_node_idx_;
// #ifdef LIMIT_ACTIVE_BLOCKS
//     AtomicU32 block_sm_offsets_[MADRONA_MWGPU_NUM_MEGAKERNEL_NUM_SMS];
// #endif
    FixedInlineArray<
            cuda::barrier<cuda::thread_scope_device>,
            MADRONA_MWGPU_MAX_BLOCKS_PER_SM
        > init_barriers_;

friend class Builder;
};

// FIXME: Compat with new CPU naming scheme
using TaskGraphNodeID = TaskGraph::NodeID;
using TaskGraphBuilder = TaskGraph::Builder;

class TaskGraphManager {
public:
    TaskGraphManager(uint32_t num_taskgraphs);

    // Create a new TaskgraphBuilder for building a task graph
    template <EnumType EnumT>
    TaskGraphBuilder & init(EnumT taskgraph_id);
    TaskGraphBuilder & init(uint32_t taskgraph_id);

    void constructGraphs();
private:
    TaskGraphBuilder *builders_;
    uint32_t num_taskgraphs_;
};

template <typename ContextT, auto Fn,
          int32_t threads_per_invocation,
          int32_t items_per_invocation,
          typename ...ComponentTs>
class CustomParallelForNode: public NodeBase {
public:
    CustomParallelForNode();

    inline void run(const int32_t invocation_idx);
    inline uint32_t numInvocations();

    static TaskGraph::NodeID addToGraph(
            TaskGraph::Builder &builder,
            Span<const TaskGraph::NodeID> dependencies);

private:
    QueryRef *query_ref_;
};

template <typename ContextT, auto Fn, typename ...ComponentTs>
using ParallelForNode =
    CustomParallelForNode<ContextT, Fn, 1, 1, ComponentTs...>;

struct ClearTmpNodeBase : NodeBase {
    ClearTmpNodeBase(uint32_t archetype_id);

    void run(int32_t);

    static TaskGraph::NodeID addToGraph(
        TaskGraph::Builder &builder,
        Span<const TaskGraph::NodeID> dependencies,
        uint32_t archetype_id);

    uint32_t archetypeID;
};

template <typename ArchetypeT>
struct ClearTmpNode : ClearTmpNodeBase {
    static TaskGraph::NodeID addToGraph(
        TaskGraph::Builder &builder,
        Span<const TaskGraph::NodeID> dependencies);
};

struct RecycleEntitiesNode : NodeBase {
    RecycleEntitiesNode();

    void run(int32_t invocation_idx);
    uint32_t numInvocations();

    static TaskGraph::NodeID addToGraph(
        TaskGraph::Builder &builder,
        Span<const TaskGraph::NodeID> dependencies);

    int32_t recycleBase;
};

struct ResetTmpAllocNode : NodeBase {
    void run(int32_t);

    static TaskGraph::NodeID addToGraph(
        TaskGraph::Builder &builder,
        Span<const TaskGraph::NodeID> dependencies);
};

struct SortArchetypeNodeBase : NodeBase {
    struct RadixSortOnesweepCustom;

    using ParentNodeT = TaskGraph::TypedDataID<SortArchetypeNodeBase>;
    struct OnesweepNode : NodeBase {
        OnesweepNode(uint32_t taskgraph_id, ParentNodeT parent,
                     int32_t pass, bool final_pass);
        uint32_t taskGraphID;
        ParentNodeT parentNode;
        int32_t passIDX;
        bool finalPass;
        uint32_t *srcKeys;
        uint32_t *dstKeys;
        int *srcVals;
        int *dstVals;

        void prepareOnesweep(int32_t invocation_idx);
        void onesweep(int32_t invocation_idx);
    };

    struct RearrangeNode : NodeBase {
        RearrangeNode(uint32_t taskgraph_id, ParentNodeT parent,
                      int32_t col_idx);
        uint32_t taskGraphID;
        TaskGraph::TypedDataID<SortArchetypeNodeBase> parentNode;
        int32_t columnIndex;
        TaskGraph::TypedDataID<RearrangeNode> nextRearrangeNode;

        void stageColumn(int32_t invocation_idx);
        void rearrangeEntities(int32_t invocation_idx);
        void rearrangeColumn(int32_t invocation_idx);
    };

    struct ClearCountNode : NodeBase {
        ClearCountNode(int32_t *world_offsets, int32_t *world_counts);

        int32_t *worldOffsets;
        int32_t *worldCounts;

        void clearCounts(int32_t invocation_idx);
    };

    SortArchetypeNodeBase(uint32_t taskgraph_id,
                          uint32_t archetype_id,
                          int32_t col_idx,
                          uint32_t *keys_col,
                          int32_t num_passes,
                          int32_t *sort_offsets,
                          int32_t *counts);

    void sortSetup(int32_t);
    void zeroBins(int32_t invocation_idx);
    void histogram(int32_t invocation_idx);
    void binScan(int32_t invocation_idx);
    void resizeTable(int32_t);
    void copyKeys(int32_t invocation_idx);
    void computeWorldCounts(int32_t invocation_idx);
    void correctWorldCounts(int32_t invocation_idx);
    void clearWorldOffsetsAndCounts(int32_t invocation_idx);
    void worldCountScan(int32_t invocation_idx);


    static TaskGraph::NodeID addToGraph(
        TaskGraph::Builder &builder,
        Span<const TaskGraph::NodeID> dependencies,
        uint32_t archetype_id,
        int32_t component_id);

    // Constant state
    uint32_t taskGraphID;
    uint32_t archetypeID;
    int32_t sortColumnIndex;
    uint32_t *keysCol;
    int32_t numPasses;
    int32_t *worldOffsets;
    int32_t *worldCounts;

    TaskGraph::TypedDataID<OnesweepNode> onesweepNodes[4];
    TaskGraph::TypedDataID<RearrangeNode> firstRearrangePassData;
    TaskGraph::TypedDataID<ClearCountNode> clearWorldCountData;

    // Per-run state
    uint32_t numRows;
    uint32_t numSortBlocks;
    uint32_t numSortThreads;
    uint32_t postBinScanThreads;
    int *indicesFinal; // Points to either indices or indicesAlt
    void *columnStaging;
    int *indices;
    int *indicesAlt;
    uint32_t *keysAlt;
    int32_t *bins;
    int32_t *lookback;
    int32_t *counters;

    static inline constexpr uint32_t num_elems_per_sort_thread_ = 2;
};

template <typename ArchetypeT, typename ComponentT>
struct SortArchetypeNode : SortArchetypeNodeBase {
    static TaskGraph::NodeID addToGraph(
        TaskGraph::Builder &builder,
        Span<const TaskGraph::NodeID> dependencies);
};

template <typename ArchetypeT>
struct CompactArchetypeNode {
    static TaskGraph::NodeID addToGraph(
        TaskGraph::Builder &builder,
        Span<const TaskGraph::NodeID> dependencies);
};

}

#include "taskgraph.inl"
