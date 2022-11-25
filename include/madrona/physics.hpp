#pragma once
#include <madrona/math.hpp>
#include <madrona/components.hpp>
#include <madrona/span.hpp>
#include <madrona/taskgraph.hpp>

namespace madrona {
namespace phys {

struct RigidBody {
    math::Vector3 invInertiaTensor;
};

namespace broadphase {

struct LeafAABB : math::AABB {
    LeafAABB(math::AABB aabb)
        : AABB(aabb)
    {}
};

struct LeafCenter : math::Vector3 {
    LeafCenter(math::Vector3 v)
        : Vector3(v)
    {}
};

struct LeafID {
    int32_t id;
};

class BVH {
public:
    BVH(CountT max_leaves);

    inline LeafID reserveLeaf();

    void rebuild();

    void refit(LeafID *leaf_ids, CountT num_moved);

    template <typename Fn>
    inline void findOverlaps(madrona::math::AABB &aabb, Fn &&fn) const;

    inline void updateLeaf(
        const madrona::base::Position &position,
        const madrona::base::Rotation &rotation,
        const LeafID &leaf_id);

    static inline void updateLeavesSystem(
        Context &ctx,
        Loc sys_loc,
        const madrona::base::Position &position,
        const madrona::base::Rotation &rotation,
        const LeafID &leaf_id);

private:
    static constexpr int32_t sentinel_ = 0xFFFF'FFFF_i32;

    inline CountT numInternalNodes(CountT num_leaves) const;

    struct Node {
        float minX[4];
        float minY[4];
        float minZ[4];
        float maxX[4];
        float maxY[4];
        float maxZ[4];
        int32_t children[4];
        int32_t parentID;
    
        inline bool isLeaf(IdxT child) const;
        inline uint32_t leafRawEntity(IdxT child) const;
    
        inline void setLeaf(IdxT child, int32_t entity_id);
        inline void setInternal(IdxT child, int32_t internal_idx);
        inline void clearChild(IdxT child);
    };

    Node *nodes_;
    CountT num_nodes_;
    const CountT num_allocated_nodes_;
    LeafAABB *leaf_aabbs_;
    LeafCenter *leaf_centers_;
    uint32_t *leaf_parents_;
    int32_t *sorted_leaves_;
    std::atomic<int32_t> num_leaves_;
    int32_t num_allocated_leaves_;
};

void registerTypes(ECSRegistry &registry);

TaskGraph::NodeID setupTasks(TaskGraph::Builder &builder,
                             Span<const TaskGraph::NodeID> deps);

}

}
}

#include "physics.inl"
