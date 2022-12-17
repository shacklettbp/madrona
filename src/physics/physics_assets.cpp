#include <madrona/physics_assets.hpp>
#include <madrona/cuda_utils.hpp>


namespace madrona {
namespace phys {

struct PhysicsLoader::Impl {
    RigidBodyMetadata *metadatas;
    math::AABB *aabbs;
    ObjectManager *mgr;
    CountT curLoadedObjs;
    CountT maxObjs;
    StorageType storageType;

    static Impl * init(StorageType storage_type, CountT max_objects)
    {
        size_t num_metadata_bytes =
            sizeof(RigidBodyMetadata) * max_objects;

        size_t num_aabb_bytes =
            sizeof(math::AABB) * max_objects;

        RigidBodyMetadata *metadata_ptr;
        math::AABB *aabb_ptr;

        ObjectManager *mgr;

        switch (storage_type) {
        case StorageType::CPU: {
            metadata_ptr =
                (RigidBodyMetadata *)malloc(num_metadata_bytes);
            aabb_ptr = (math::AABB *)malloc(num_aabb_bytes);
            mgr = new ObjectManager {
                metadata_ptr,
                aabb_ptr,
            };
        } break;
        case StorageType::CUDA: {
            metadata_ptr =
                (RigidBodyMetadata *)cu::allocGPU(num_metadata_bytes);
            aabb_ptr = (math::AABB *)cu::allocGPU(num_aabb_bytes);

            mgr = (ObjectManager *)cu::allocGPU(sizeof(ObjectManager));

            ObjectManager local {
                metadata_ptr,
                aabb_ptr,
            };

            REQ_CUDA(cudaMemcpy(mgr, &local, sizeof(ObjectManager),
                                cudaMemcpyHostToDevice));
        } break;
        default: __builtin_unreachable();
        }

        return new Impl {
            .metadatas = metadata_ptr,
            .aabbs = aabb_ptr,
            .mgr = mgr,
            .curLoadedObjs = 0,
            .maxObjs = max_objects,
            .storageType = storage_type,
        };
    }
};

PhysicsLoader::PhysicsLoader(StorageType storage_type, CountT max_objects)
    : impl_(Impl::init(storage_type, max_objects))
{}

PhysicsLoader::~PhysicsLoader()
{
    if (impl_ == nullptr) {
        return;
    }

    switch (impl_->storageType) {
    case StorageType::CPU: {
        delete impl_->mgr;
        free(impl_->aabbs);
        free(impl_->metadatas);
    } break;
    case StorageType::CUDA: {
        cu::deallocGPU(impl_->mgr);
        cu::deallocGPU(impl_->aabbs);
        cu::deallocGPU(impl_->metadatas);
    } break;
    }
}

PhysicsLoader::PhysicsLoader(PhysicsLoader &&o) = default;

CountT PhysicsLoader::loadObjects(const RigidBodyMetadata *metadatas,
                                  const math::AABB *aabbs,
                                  CountT num_objs)
{
    CountT cur_offset = impl_->curLoadedObjs;
    impl_->curLoadedObjs += num_objs;
    assert(impl_->curLoadedObjs <= impl_->maxObjs);

    size_t num_metadata_bytes = sizeof(RigidBodyMetadata) * num_objs;
    size_t num_aabb_bytes = sizeof(math::AABB) * num_objs;

    RigidBodyMetadata *metadatas_dst = &impl_->metadatas[cur_offset];
    math::AABB *aabbs_dst = &impl_->aabbs[cur_offset];
    
    switch (impl_->storageType) {
    case StorageType::CPU: {
        memcpy(metadatas_dst, metadatas, num_metadata_bytes);
        memcpy(aabbs_dst, aabbs, num_aabb_bytes);
    } break;
    case StorageType::CUDA: {
        cudaMemcpy(metadatas_dst, metadatas, num_metadata_bytes,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(aabbs_dst, aabbs, num_aabb_bytes,
                   cudaMemcpyHostToDevice);
    } break;
    default: __builtin_unreachable();
    }

    return cur_offset;
}

ObjectManager & PhysicsLoader::getObjectManager()
{
    return *impl_->mgr;
}

}
}
