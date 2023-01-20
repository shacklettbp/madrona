#pragma once

#include <madrona/span.hpp>
#include <madrona/query.hpp>

#include <madrona/state.hpp>

#include "mw_gpu/const.hpp"
#include "mw_gpu/worker_init.hpp"
#include "mw_gpu/megakernel_consts.hpp"

#include <cuda/barrier>
#include <cuda/std/tuple>

#define LIMIT_ACTIVE_THREADS
// #define LIMIT_ACTIVE_BLOCKS
// #define FETCH_MULTI_INVOCATIONS

namespace madrona {

class TaskGraph;

namespace mwGPU {
    inline TaskGraph * getTaskGraph();
}

struct NodeBase {
    uint32_t numDynamicInvocations;
};

class TaskGraph {
private:
    static inline constexpr uint32_t maxNodeDataBytes = 128;
    struct alignas(maxNodeDataBytes) NodeData {
        char userData[maxNodeDataBytes];
    };
    static_assert(sizeof(NodeData) == 128);

    struct Node {
        uint32_t dataIDX;
        uint32_t fixedCount;
        uint32_t funcID;
        uint32_t numChildren;
        uint32_t numThreadsPerInvocation;

        std::atomic_uint32_t curOffset;
        std::atomic_uint32_t numRemaining;
        std::atomic_uint32_t totalNumInvocations;
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
        Builder(int32_t max_nodes,
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
        uint32_t num_dependencies_;
        uint32_t num_worlds_;
    };

    enum class WorkerState {
        Run,
        PartialRun,
        Loop,
        Exit,
    };

    TaskGraph(const TaskGraph &) = delete;
    ~TaskGraph();

    void init();

    WorkerState getWork(NodeBase **node_data,
                        uint32_t *run_func_id,
                        int32_t *run_offset);

    void finishWork();

    static inline WorldBase * getWorld(int32_t world_idx);

    template <typename ContextT>
    static ContextT makeContext(WorldID world_id);

    template <typename NodeT>
    NodeT & getNodeData(TypedDataID<NodeT> data_id);

    struct BlockState;
private:
    template <typename ContextT, bool> struct WorldTypeExtract;

    TaskGraph(Node *nodes, uint32_t num_nodes,
              NodeData *node_datas);

    inline void setBlockState();
    inline uint32_t computeNumInvocations(Node &node);

    Node *sorted_nodes_;
    uint32_t num_nodes_;
    NodeData *node_datas_;
    std::atomic_uint32_t cur_node_idx_;
    static uint32_t const num_SMs_ = MADRONA_MWGPU_NUM_MEGAKERNEL_BLOCKS / consts::numMegakernelBlocksPerSM;
#ifdef LIMIT_ACTIVE_BLOCKS
    std::atomic_uint32_t block_sm_offsets_[num_SMs_];
#endif
    cuda::barrier<cuda::thread_scope_device> init_barrier_;

friend class Builder;
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

struct CompactArchetypeNodeBase : NodeBase {
    CompactArchetypeNodeBase(uint32_t archetype_id);

    void run(int32_t invocation_idx);
    uint32_t numInvocations();

    static TaskGraph::NodeID addToGraph(
        TaskGraph::Builder &builder,
        Span<const TaskGraph::NodeID> dependencies,
        uint32_t archetype_id);

    uint32_t archetypeID;
};

template <typename ArchetypeT>
struct CompactArchetypeNode : CompactArchetypeNodeBase {
    static TaskGraph::NodeID addToGraph(
        TaskGraph::Builder &builder,
        Span<const TaskGraph::NodeID> dependencies);
};

struct SortArchetypeNodeBase : NodeBase {
    struct RadixSortOnesweepCustom;

    using ParentNodeT = TaskGraph::TypedDataID<SortArchetypeNodeBase>;
    struct OnesweepNode : NodeBase {
        OnesweepNode(ParentNodeT parent, int32_t pass, bool final_pass);
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
        RearrangeNode(ParentNodeT parent, int32_t col_idx);
        TaskGraph::TypedDataID<SortArchetypeNodeBase> parentNode;
        int32_t columnIndex;
        TaskGraph::TypedDataID<RearrangeNode> nextRearrangeNode;

        void stageColumn(int32_t invocation_idx);
        void rearrangeEntities(int32_t invocation_idx);
        void rearrangeColumn(int32_t invocation_idx);
    };

    SortArchetypeNodeBase(uint32_t archetype_id,
                          int32_t col_idx,
                          uint32_t *keys_col,
                          int32_t num_passes);

    void sortSetup(int32_t);
    void zeroBins(int32_t invocation_idx);
    void histogram(int32_t invocation_idx);
    void binScan(int32_t invocation_idx);
    void resizeTable(int32_t);
    void copyKeys(int32_t invocation_idx);

    static TaskGraph::NodeID addToGraph(
        TaskGraph::Builder &builder,
        Span<const TaskGraph::NodeID> dependencies,
        uint32_t archetype_id,
        int32_t component_id);

    // Constant state
    uint32_t archetypeID;
    int32_t sortColumnIndex;
    uint32_t *keysCol;
    int32_t numPasses;

    TaskGraph::TypedDataID<OnesweepNode> onesweepNodes[4];
    TaskGraph::TypedDataID<RearrangeNode> firstRearrangePassData;

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

}

#include "taskgraph.inl"
