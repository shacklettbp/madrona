#pragma once

#include <madrona/physics.hpp>

namespace madrona {
namespace phys {

class PhysicsLoader {
public:
    enum class StorageType {
        CPU,
        CUDA,
    };

    PhysicsLoader(StorageType storage_type, CountT max_objects);
    ~PhysicsLoader();

    ObjectManager * getObjectManager() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    ObjectManager *mgr_;
    CountT cur_loaded_objs_;
    CountT max_objs_;
    StorageType storage_type_;
};

}
}
