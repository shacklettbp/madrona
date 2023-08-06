#pragma once

namespace madrona::phys {

namespace geometry {

template <typename Fn>
void HalfEdgeMesh::iterateFaceIndices(uint32_t face, Fn &&fn) const
{
    uint32_t hedge_idx = faceBaseHalfEdges[face];
    uint32_t start = hedge_idx;

    do {
        const HalfEdge &hedge = halfEdges[hedge_idx];
        fn(hedge.rootVertex);

        hedge_idx = hedge.next;
    } while (hedge_idx != start);
}

uint32_t HalfEdgeMesh::twinIDX(uint32_t half_edge_id) const
{
    if ((half_edge_id & 1) == 1) {
        return half_edge_id & 0xFFFF'FFFE;
    } else {
        return half_edge_id | 1;
    }
}

uint32_t HalfEdgeMesh::numEdges() const
{
    return numHalfEdges / 2;
}

uint32_t HalfEdgeMesh::edgeToHalfEdge(uint32_t edge_id) const
{
    return edge_id * 2;
}

}

namespace broadphase {

LeafID BVH::reserveLeaf(Entity e, base::ObjectID obj_id)
{
    int32_t leaf_idx = num_leaves_.fetch_add_relaxed(1);
    assert(leaf_idx < num_allocated_leaves_);

    leaf_entities_[leaf_idx] = e;
    leaf_obj_ids_[leaf_idx] = obj_id;

    return LeafID {
        leaf_idx,
    };
}

math::AABB BVH::getLeafAABB(LeafID leaf_id) const
{
    return leaf_aabbs_[leaf_id.id];
}

template <typename Fn>
void BVH::findOverlaps(const math::AABB &aabb, Fn &&fn) const
{
    int32_t stack[32];
    stack[0] = 0;
    CountT stack_size = 1;

    while (stack_size > 0) {
        int32_t node_idx = stack[--stack_size];
        const Node &node = nodes_[node_idx];
        for (int i = 0; i < 4; i++) {
            if (!node.hasChild(i)) {
                continue; // Technically this could be break?
            };

            madrona::math::AABB child_aabb {
                /* .pMin = */ {
                    node.minX[i],
                    node.minY[i],
                    node.minZ[i],
                },
                /* .pMax = */ {
                    node.maxX[i],
                    node.maxY[i],
                    node.maxZ[i],
                },
            };

            if (aabb.overlaps(child_aabb)) {
                if (node.isLeaf(i)) {
                    Entity e = leaf_entities_[node.leafIDX(i)];
                    fn(e);
                } else {
                    stack[stack_size++] = node.children[i];
                }
            }
        }
    }
}

template <typename Fn>
void BVH::findOverlapsForLeaf(LeafID leaf_id, Fn &&fn) const
{
    math::AABB leaf_aabb = leaf_aabbs_[leaf_id.id];
    findOverlaps(leaf_aabb, std::forward<Fn>(fn));
}

void BVH::rebuildOnUpdate()
{
    force_rebuild_ = true;
}

void BVH::clearLeaves()
{
    num_leaves_.store_relaxed(0);
}

bool BVH::Node::isLeaf(CountT child) const
{
    return children[child] & 0x80000000;
}

int32_t BVH::Node::leafIDX(CountT child) const
{
    return children[child] & ~0x80000000;
}

void BVH::Node::setLeaf(CountT child, int32_t idx)
{
    children[child] = 0x80000000 | idx;
}

void BVH::Node::setInternal(CountT child, int32_t internal_idx)
{
    children[child] = internal_idx;
}

bool BVH::Node::hasChild(CountT child) const
{
    return children[child] != sentinel_;
}

void BVH::Node::clearChild(CountT child)
{
    children[child] = sentinel_;
}

}

JointConstraint JointConstraint::setupFixed(
    Entity e1, Entity e2,
    math::Quat attach_rot1, math::Quat attach_rot2,
    math::Vector3 r1, math::Vector3 r2,
    float separation)
{
    return JointConstraint {
        .e1 = e1,
        .e2 = e2,
        .type = JointConstraint::Type::Fixed,
        .fixed = {
            .attachRot1 = attach_rot1,
            .attachRot2 = attach_rot2,
            .separation = separation,
        },
        .r1 = r1,
        .r2 = r2,
    };
}

JointConstraint JointConstraint::setupHinge(
    Entity e1, Entity e2,
    math::Vector3 a1_local, math::Vector3 a2_local,
    math::Vector3 b1_local, math::Vector3 b2_local,
    math::Vector3 r1, math::Vector3 r2)
{
    return JointConstraint {
        .e1 = e1,
        .e2 = e2,
        .type = JointConstraint::Type::Hinge,
        .hinge = {
            .a1Local = a1_local,
            .a2Local = a2_local,
            .b1Local = b1_local,
            .b2Local = b2_local,
        },
        .r1 = r1,
        .r2 = r2,
    };
}

template <typename Fn>
void RigidBodyPhysicsSystem::findEntitiesWithinAABB(Context &ctx,
                                                    math::AABB aabb,
                                                    Fn &&fn)
{
    using namespace madrona::base;
    using namespace madrona::math;

    auto &bvh = ctx.singleton<broadphase::BVH>();

    bvh.findOverlaps(aabb, [&](Entity e) {
        bool overlap = RigidBodyPhysicsSystem::checkEntityAABBOverlap(
            ctx, aabb, e);
        if (overlap) {
            fn(e);
        }
    });
}

}
