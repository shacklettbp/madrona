#pragma once
#include <madrona/math.hpp>
#include <madrona/components.hpp>
#include <madrona/span.hpp>
#include <madrona/context.hpp>

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
    BVH(CountT max_nodes);

    void build(Context &ctx, Entity *entities,
               CountT num_entities);

    void update(Context &ctx,
                Entity *added_entities,
                CountT num_added_entities,
                Entity *removed_entities,
                CountT num_removed_entities,
                Entity *moved_entities,
                CountT num_moved_entities);

    template <typename Fn>
    inline void findOverlaps(madrona::math::AABB &aabb, Fn &&fn) const;

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
    
        inline bool isLeaf(IdxT child) const;
        inline uint32_t leafRawEntity(IdxT child) const;
    
        inline void setLeaf(IdxT child, int32_t entity_id);
        inline void setInternal(IdxT child, int32_t internal_idx);
        inline void clearChild(IdxT child);
    };

    Node *nodes_;
    CountT num_nodes_;
    const CountT num_allocated_nodes_;
};

}

}
}

#include "physics.inl"
