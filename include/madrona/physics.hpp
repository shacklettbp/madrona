#pragma once
#include <madrona/math.hpp>
#include <madrona/components.hpp>
#include <madrona/system.hpp>
#include <madrona/span.hpp>
#include <madrona/context.hpp>

namespace madrona {
namespace phys {

struct BroadphaseAABB : math::AABB {
    BroadphaseAABB(math::AABB aabb)
        : AABB(aabb)
    {}
};

struct RigidBody {
    math::Vector3 invInertiaTensor;
};


class BroadphaseBVH {
public:
    BroadphaseBVH(CountT max_nodes);

    void build(Context &ctx, Span<Entity> added_entities);
    void update(Context &ctx,
                Span<Entity> added_entities,
                Span<Entity> removed_entities,
                Span<Entity> moved_entities);

    template <typename Fn>
    inline void findOverlaps(madrona::math::AABB &aabb, Fn &&fn);

private:
    static constexpr int32_t sentinel_ = 0xFFFFFFFF;

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


class BroadphaseSystem : ParallelForSystem<BroadphaseSystem,
                                           Entity, BroadphaseAABB> {
    void run();
};

}
}

#include "physics.inl"
