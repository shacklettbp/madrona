#pragma once

#include <madrona/physics.hpp>
#include <madrona/physics_assets.hpp>

#include <madrona/exec_mode.hpp>

namespace madrona::phys {

class PhysicsLoader {
public:
    PhysicsLoader(ExecMode exec_mode, CountT max_objects);
    ~PhysicsLoader();
    PhysicsLoader(PhysicsLoader &&o);

    CountT loadRigidBodies(const RigidBodyAssets &assets);

    ObjectManager & getObjectManager();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
