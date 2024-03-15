#pragma once

namespace madrona::phys {

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
