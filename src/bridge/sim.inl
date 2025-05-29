namespace madMJX {

template <typename ArchetypeT>
madrona::Entity Engine::makeRenderableEntity()
{
    Entity e = makeEntity<ArchetypeT>();
    madrona::render::RenderingSystem::makeEntityRenderable(*this, e);
    
    return e;
}

inline void Engine::destroyRenderableEntity(Entity e)
{
    madrona::render::RenderingSystem::cleanupRenderableEntity(*this, e);
    destroyEntity(e);
}

}
