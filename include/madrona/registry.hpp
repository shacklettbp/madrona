/*
 * Copyright 2021-2023 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#include <madrona/state.hpp>

namespace madrona {

// ECSRegistry is passed to the application's registerTypes function.
// Application code uses this class to register all its custom 
// components and archetypes, and passes it to any libraries it uses
// so they can also register their types.
class ECSRegistry {
public:
    ECSRegistry(StateManager *state_mgr, void **export_ptrs);
 
    // Use as follows to register MyComponent:
    // registry.registerComponent<MyComponent>();
    template <typename ComponentT>
    void registerComponent(uint32_t num_bytes = 0);

    // Use as follows to register MyArchetype:
    // registry.registerComponent<MyArchetype>();
    template <typename ArchetypeT>
    void registerArchetype();

    // Use as follows to register an archetype but for which we want to specify
    // special properties for certain components:
    // registry.registerComponent<MyArchetype>(
    //     ComponentSelector<MyComponentA, MyComponentB>(FlagsA, FlagsB);
    template <typename ArchetypeT, typename... MetadataComponentTs>
    void registerArchetype(
        ComponentMetadataSelector<MetadataComponentTs...> component_metadatas,
        ArchetypeFlags archetype_flags,
        CountT max_num_entities_per_world = 0);

    template <typename BundleT>
    void registerBundle();

    template <typename AliasT, typename BundleT>
    void registerBundleAlias();

    // Register a singleton component. Note that you should pass the desired
    // component type to this function, not an archetype (singletons implicitly
    // create an archetype with 1 component).
    template <typename SingletonT>
    void registerSingleton();

    // Export ComponentT of ArchetypeT for use by code outside the ECS,
    // such as learning. The exported pointer can be retrieved from the CPU or
    // GPU backend's getExported() function by passing the same value of 'slot'
    // as used in this call. Make sure the the numExportedBuffers config
    // parameter is set appropriately for the backend to ensure there are
    // enough slots!
    template <typename ArchetypeT, typename ComponentT>
    void exportColumn(int32_t slot);

    // Same as exportColumn, directly export the SingletonT component.
    template <typename SingletonT>
    void exportSingleton(int32_t slot);

    template <typename ArchetypeT, typename ComponentT, EnumType EnumT>
    void exportColumn(EnumT slot);
    template <typename SingletonT, EnumType EnumT>
    void exportSingleton(EnumT slot);

private:
    StateManager *state_mgr_;
    void **export_ptrs_;
};

}

#include "registry.inl"
