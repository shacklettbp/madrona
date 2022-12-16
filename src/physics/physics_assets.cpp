#include <madrona/physics_assets.hpp>

namespace madrona {
namespace phys {

struct PhysicsLoader::Impl {
    ObjectManager *mgr_;
    CountT cur_loaded_objs_;
    CountT max_objs_;
    StorageType storage_type_;

    static Impl * init(StorageType storage_type, CountT max_objects)
    {
        (void)storage_type;
        (void)max_objects;
        return nullptr;
    }
};

PhysicsLoader::PhysicsLoader(StorageType storage_type, CountT max_objects)
    : impl_(Impl::init(storage_type, max_objects))
{}

ObjectManager * PhysicsLoader::getObjectManager() const
{
    return impl_->mgr_;
}

}
}
