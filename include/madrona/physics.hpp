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
};

struct CollisionEvent {
    Entity a;
    Entity b;
};

struct RigidBodyPhysicsSystem {
    static void init(Context &ctx, CountT max_dynamic_objects,
                     CountT max_contacts_per_step);
    static void registerTypes(ECSRegistry &registry);
    static TaskGraph::NodeID setupTasks(TaskGraph::Builder &builder,
                                        Span<const TaskGraph::NodeID> deps);

    static TaskGraph::NodeID setupCleanupTasks(TaskGraph::Builder &builder,
        Span<const TaskGraph::NodeID> deps);

};

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
    inline void findOverlaps(const math::AABB &aabb, Fn &&fn) const;

    void updateLeaf(Entity e,
                    LeafID leaf_id,
                    const CollisionAABB &obj_aabb);

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
        inline int32_t leafIDX(IdxT child) const;
    
        inline void setLeaf(IdxT child, int32_t idx);
        inline void setInternal(IdxT child, int32_t internal_idx);
        inline bool hasChild(IdxT child) const;
        inline void clearChild(IdxT child);
    };

    Node *nodes_;
    CountT num_nodes_;
    const CountT num_allocated_nodes_;
    math::AABB *leaf_aabbs_;
    math::Vector3 *leaf_centers_;
    uint32_t *leaf_parents_;
    Entity *leaf_entities_;
    int32_t *sorted_leaves_;
    std::atomic<int32_t> num_leaves_;
    int32_t num_allocated_leaves_;
};

}

}
}

#include "physics.inl"
