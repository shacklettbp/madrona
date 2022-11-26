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

struct CollisionAABB : math::AABB {
    CollisionAABB(math::AABB aabb)
        : AABB(aabb)
    {}

    CollisionAABB(const base::Position &pos,
                  const base::Rotation &rot);
    
    static TaskGraph::NodeID setupTasks(TaskGraph::Builder &builder,
                                        Span<const TaskGraph::NodeID> deps);
};

void registerTypes(ECSRegistry &registry);

namespace broadphase {

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
    inline void findOverlaps(math::AABB &aabb, Fn &&fn) const;

    void updateLeaf(LeafID leaf_id, const CollisionAABB &obj_aabb);

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
    math::AABB *leaf_aabbs_;
    math::Vector3 *leaf_centers_;
    uint32_t *leaf_parents_;
    int32_t *sorted_leaves_;
    std::atomic<int32_t> num_leaves_;
    int32_t num_allocated_leaves_;
};

class System {
public:
    static void registerTypes(ECSRegistry &registry);

    static void init(Context &ctx, CountT max_num_leaves);

    static TaskGraph::NodeID setupTasks(TaskGraph::Builder &builder,
                                        Span<const TaskGraph::NodeID> deps);

private:
    static void updateLeavesEntry(
        Context &ctx,
        const LeafID &leaf_id,
        const CollisionAABB &aabb);

    static void updateBVHEntry(
        Context &ctx, BVH &bvh);

    static void findOverlappingEntry(
        Context &ctx, const CollisionAABB &obj_aabb);

};


}

}
}

#include "physics.inl"
