namespace madrona::phys::cv {
template <typename ArchetypeT, typename ComponentT>
inline ComponentT * getRows(StateManager *state_mgr, uint32_t world_id)
{
#ifdef MADRONA_GPU_MODE
    (void)world_id;
    return state_mgr->getArchetypeComponent<ArchetypeT, ComponentT>();
#else
    return state_mgr->getWorldComponents<ArchetypeT, ComponentT>(world_id);
#endif
}
}
