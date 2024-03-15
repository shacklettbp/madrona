#pragma once

#include <madrona/math.hpp>
#include <madrona/context.hpp>

namespace madrona::phys {

struct ObjectManager;

}

namespace madrona::phys::broadphase {

struct LeafID {
    int32_t id;
};

class BVH {
public:
    BVH(const ObjectManager *obj_mgr,
        CountT max_leaves,
        float leaf_velocity_expansion,
        float leaf_accel_expansion);

    inline LeafID reserveLeaf(Entity e, base::ObjectID obj_id);
    inline math::AABB getLeafAABB(LeafID leaf_id) const;

    template <typename Fn>
    inline void findIntersecting(const math::AABB &aabb, Fn &&fn) const;

    template <typename Fn>
    inline void findLeafIntersecting(LeafID leaf_id, Fn &&fn) const;

    Entity traceRay(math::Vector3 o,
                    math::Vector3 d,
                    float *out_hit_t,
                    math::Vector3 *out_hit_normal,
                    float t_max = float(INFINITY));

    void updateLeafPosition(LeafID leaf_id,
                            const math::Vector3 &pos,
                            const math::Quat &rot,
                            const math::Diag3x3 &scale,
                            const math::Vector3 &linear_vel,
                            const math::AABB &obj_aabb);

    math::AABB expandLeaf(LeafID leaf_id,
                          const math::Vector3 &linear_vel);

    void refitLeaf(LeafID leaf_id, const math::AABB &leaf_aabb);

    inline void rebuildOnUpdate();
    void updateTree();

    inline void clearLeaves();

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

    // FIXME: evaluate whether storing this in-line in the tree
    // makes sense or if we should force a lookup through the entity ID
    struct LeafTransform {
        math::Vector3 pos;
        math::Quat rot;
        math::Diag3x3 scale;
    };

    inline CountT numInternalNodes(CountT num_leaves) const;

    void rebuild();
    void refit(LeafID *leaf_ids, CountT num_moved);

    bool traceRayIntoLeaf(int32_t leaf_idx,
                          math::Vector3 world_ray_o,
                          math::Vector3 world_ray_d,
                          float t_min,
                          float t_max,
                          float *hit_t,
                          math::Vector3 *hit_normal);

    Node *nodes_;
    CountT num_nodes_;
    const CountT num_allocated_nodes_;
    Entity *leaf_entities_;
    const ObjectManager *obj_mgr_;
    base::ObjectID *leaf_obj_ids_;
    math::AABB *leaf_aabbs_; // FIXME: remove this, it's duplicated data
    LeafTransform  *leaf_transforms_;
    uint32_t *leaf_parents_;
    int32_t *sorted_leaves_;
    AtomicI32 num_leaves_;
    int32_t num_allocated_leaves_;
    float leaf_velocity_expansion_;
    float leaf_accel_expansion_;
    bool force_rebuild_;
};

}

#include "broadphase.inl"
