namespace madrona::render {
    
namespace RenderingSystem {

template <typename OutputT>
OutputT *getRenderOutput(Context &ctx,
                         const RenderCamera &camera,
                         uint32_t output_size)
{
#if MADRONA_GPU_MODE
    // What we need from this is the Loc because the index of the
    // PerspectiveCameraData struct is what determines the index of this
    // entity's view output.
    Entity camera_entity = camera.cameraEntity;
    Loc cam_entity_loc = ctx.loc(camera_entity);

    StateManager *mgr = mwGPU::getStateManager();

    uint8_t *output_ptr = (uint8_t *)mgr->getArchetypeComponent<
        RaycastOutputArchetype, OutputT>();

    output_ptr += output_size * cam_entity_loc.row;

    return (OutputT *)output_ptr;
#else
    return nullptr;
#endif
}
    
}

}
