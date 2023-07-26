#pragma once

namespace madrona {

template <typename ComponentT>
void ECSRegistry::registerComponent()
{
    state_mgr_->registerComponent<ComponentT>();
}

template <typename ArchetypeT>
void ECSRegistry::registerArchetype()
{
    state_mgr_->registerArchetype<ArchetypeT>();
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

}
