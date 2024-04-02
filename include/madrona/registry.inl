#pragma once

namespace madrona {

template <typename ComponentT>
void ECSRegistry::registerComponent(uint32_t num_bytes)
{
    state_mgr_->registerComponent<ComponentT>(num_bytes);
}

template <typename ArchetypeT>
void ECSRegistry::registerArchetype()
{
    state_mgr_->registerArchetype<ArchetypeT>(
        ComponentMetadataSelector {}, ArchetypeFlags::None, 0);
}

template <typename ArchetypeT, typename... MetadataComponentTs>
void ECSRegistry::registerArchetype(
        ComponentMetadataSelector<MetadataComponentTs...> component_metadatas,
        ArchetypeFlags archetype_flags,
        CountT max_num_entities_per_world)
{
    state_mgr_->registerArchetype<ArchetypeT>(
        component_metadatas, archetype_flags, max_num_entities_per_world);
}

template <typename BundleT>
void ECSRegistry::registerBundle()
{
    state_mgr_->registerBundle<BundleT>();
}

template <typename AliasT, typename BundleT>
void ECSRegistry::registerBundleAlias()
{
    state_mgr_->registerBundleAlias<AliasT, BundleT>();
}

template <typename SingletonT>
void ECSRegistry::registerSingleton()
{
    state_mgr_->registerSingleton<SingletonT>();
}

template <typename ArchetypeT, typename ComponentT>
void ECSRegistry::exportColumn(int32_t slot)
{
    export_ptrs_[slot] = state_mgr_->exportColumn<ArchetypeT, ComponentT>();
}

template <typename SingletonT>
void ECSRegistry::exportSingleton(int32_t slot)
{
    export_ptrs_[slot] = state_mgr_->exportSingleton<SingletonT>();
}

template <typename ArchetypeT, typename ComponentT, EnumType EnumT>
void ECSRegistry::exportColumn(EnumT slot)
{
    exportColumn<ArchetypeT, ComponentT>(static_cast<uint32_t>(slot));
}

template <typename SingletonT, EnumType EnumT>
void ECSRegistry::exportSingleton(EnumT slot)
{
    exportSingleton<SingletonT>(static_cast<uint32_t>(slot));
}

}
