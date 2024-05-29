namespace madrona::render {
    
namespace RenderingSystem {

template <typename OutputT>
OutputT *getRenderOutput(Context &ctx,
                         const RenderCamera &camera,
                         uint32_t output_size)
{
#if MADRONA_GPU_MODE
    // What determines where the output is, is the rowIDX set in the 
    // PerspectiveCameraData struct.
    Entity camera_entity = camera.cameraEntity;
    uint32_t offset = ctx.get<PerspectiveCameraData>(camera_entity).rowIDX;

    StateManager *mgr = mwGPU::getStateManager();

    uint8_t *output_ptr = (uint8_t *)mgr->getArchetypeComponent<
        RaycastOutputArchetype, OutputT>();

    output_ptr += output_size * offset;

    return (OutputT *)output_ptr;
#else
    return nullptr;
#endif
}
    
}

}
