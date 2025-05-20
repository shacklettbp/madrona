#include <madrona/render/ecs.hpp>

#include <madrona/bvh.hpp>

extern "C" __global__ void initBVHParams(madrona::BVHParams *params,
                                         uint32_t num_worlds,
                                         void *internal_data,
                                         void *bvhs,
                                         uint32_t num_bvhs,
                                         void *timings,
                                         void *materials,
                                         void *textures,
                                         float near_sphere,
                                         uint32_t num_sms,
                                         uint32_t sm_shared_memory)
{
    using namespace madrona;
    using namespace madrona::render;

    // Need to get the pointers to instances, views, offsets, etc...
    StateManager *mgr = mwGPU::getStateManager();
    mwGPU::HostAllocator *host_alloc = mwGPU::getHostAllocator();
    mwGPU::TmpAllocator *tmp_alloc = &mwGPU::TmpAllocator::get();
    mwGPU::HostPrint *host_print = 
        (mwGPU::HostPrint *)mwGPU::GPUImplConsts::get().hostPrintAddr;
    uint32_t raycast_rgbd = mwGPU::GPUImplConsts::get().raycastRGBD;

    params->numWorlds = num_worlds;

    params->instances = mgr->getArchetypeComponent<
        RenderableArchetype, InstanceData>();

    params->views = mgr->getArchetypeComponent<
        RenderCameraArchetype, PerspectiveCameraData>();

    params->lights = mgr->getArchetypeComponent<
        LightArchetype, LightDesc>();

    params->instanceOffsets = (int32_t *)mgr->getArchetypeWorldOffsets<
        RenderableArchetype>();

    params->instanceCounts = (int32_t *)mgr->getArchetypeWorldCounts<
        RenderableArchetype>();

    params->aabbs = (TLBVHNode *)mgr->getArchetypeComponent<
        RenderableArchetype, TLBVHNode>();

    params->viewOffsets = (int32_t *)mgr->getArchetypeWorldOffsets<
        RenderCameraArchetype>();

    params->viewCounts = (int32_t *)mgr->getArchetypeWorldCounts<
        RenderCameraArchetype>();

    params->lightOffsets = (int32_t *)mgr->getArchetypeWorldOffsets<
        LightArchetype>();

    params->lightCounts = (int32_t *)mgr->getArchetypeWorldCounts<
        LightArchetype>();

    params->mortonCodes = (uint32_t *)mgr->getArchetypeComponent<
        RenderableArchetype, MortonCode>();

    params->bvhs = (MeshBVH *)bvhs;

    params->timingInfo = (KernelTimingInfo *)timings;

    params->rgbOutput = (void *)mgr->getArchetypeComponent<
        RaycastOutputArchetype, render::RGBOutputBuffer>();

    params->depthOutput = (void *)mgr->getArchetypeComponent<
        RaycastOutputArchetype, render::DepthOutputBuffer>();

    params->renderOutputWidth = 
        mwGPU::GPUImplConsts::get().raycastOutputWidth;
    params->renderOutputHeight = 
        mwGPU::GPUImplConsts::get().raycastOutputHeight;

    params->internalData = (BVHInternalData *)internal_data;

    params->hostAllocator = (void *)host_alloc;
    params->tmpAllocator = (void *)tmp_alloc;
    params->hostPrintAddr = (void *)host_print;

    params->materials = (Material *)materials;

    params->textures = (cudaTextureObject_t *)textures;

    params->nearSphere = near_sphere;

    params->raycastRGBD = raycast_rgbd;

    params->numSMs = num_sms;
    params->smSharedMemory = sm_shared_memory;
}
