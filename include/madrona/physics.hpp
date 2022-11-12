#pragma once
#include <madrona/math.hpp>
#include <madrona/components.hpp>
#include <madrona/system.hpp>

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

class BroadphaseSystem : ParallelForSystem<BroadphaseSystem,
                                           Entity, BroadphaseAABB> {
    void run();
};

}
}
