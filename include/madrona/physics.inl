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

namespace PhysicsSystem {

template <typename Fn>
void findEntitiesWithinAABB(Context &ctx,
                                                    math::AABB aabb,
                                                    Fn &&fn)
{
    using namespace madrona::base;
    using namespace madrona::math;

    auto &bvh = ctx.singleton<broadphase::BVH>();

    bvh.findIntersecting(aabb, [&](Entity e) {
        bool overlap = checkEntityAABBOverlap(
            ctx, aabb, e);
        if (overlap) {
            fn(e);
        }
    });
}

}

}
