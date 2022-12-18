#pragma once
#include <madrona/math.hpp>
#include <madrona/components.hpp>
#include <madrona/span.hpp>
#include <madrona/taskgraph.hpp>

namespace madrona {
namespace phys {

struct Velocity : public math::Vector3 {
    Velocity(math::Vector3 v)
        : Vector3(v)
    {}
};

struct CollisionAABB : math::AABB {
    inline CollisionAABB(math::AABB aabb)
        : AABB(aabb)
    {}
};

struct CollisionEvent {
    Entity a;
    Entity b;
};

// Per object state
struct RigidBodyMetadata {
    math::Vector3 invInertiaTensor;
};

struct CollisionPrimitive {
    enum class Type : uint32_t {
        Sphere = 1 << 0,
        Hull = 1 << 1,
        Plane = 1 << 2,
    };

    struct Sphere {
        float radius;
    };

    struct Hull {
        // TODO
    };

    struct Plane {};

    Type type;
    union {
        Sphere sphere;
        Plane plane;
        Hull hull;
    };
};

struct ObjectManager {
    RigidBodyMetadata *metadata;
    math::AABB *aabbs;
    CollisionPrimitive *primitives;
};

namespace broadphase {

struct LeafID {
    int32_t id;
};

class BVH {
public:
    BVH(CountT max_leaves);

    inline LeafID reserveLeaf();

    template <typename Fn>
    inline void findOverlaps(const math::AABB &aabb, Fn &&fn) const;

    void updateLeaf(Entity e,
                    LeafID leaf_id,
                    const CollisionAABB &obj_aabb);

    inline void rebuildOnUpdate();
    void updateTree();

private:
    static constexpr int32_t sentinel_ = 0xFFFF'FFFF_i32;

    struct Node {
        float minX[4];
        float minY[4];
        float minZ[4];
        float maxX[4];
        float maxY[4];
        float maxZ[4];
        int32_t children[4];
        int32_t parentID;
    
        inline bool isLeaf(CountT child) const;
        inline int32_t leafIDX(CountT child) const;
    
        inline void setLeaf(CountT child, int32_t idx);
        inline void setInternal(CountT child, int32_t internal_idx);
        inline bool hasChild(CountT child) const;
        inline void clearChild(CountT child);
    };

    inline CountT numInternalNodes(CountT num_leaves) const;

    void rebuild();
    void refit(LeafID *leaf_ids, CountT num_moved);

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
    bool force_rebuild_;
};

}

namespace solver {

struct InstanceState {
    math::Vector3 prevPosition;
};

}

struct RigidBodyPhysicsSystem {
    static void init(Context &ctx,
                     ObjectManager *obj_mgr,
                     float delta_t,
                     CountT num_substeps,
                     CountT max_dynamic_objects,
                     CountT max_contacts_per_step);

    static void reset(Context &ctx);
    static broadphase::LeafID registerObject(Context &ctx);

    static void registerTypes(ECSRegistry &registry);
    static TaskGraph::NodeID setupTasks(TaskGraph::Builder &builder,
                                        Span<const TaskGraph::NodeID> deps,
                                        CountT num_substeps);

    static TaskGraph::NodeID setupCleanupTasks(TaskGraph::Builder &builder,
        Span<const TaskGraph::NodeID> deps);

};


}
}

#include "physics.inl"
