/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include <madrona/state.hpp>

namespace madrona {

StateManager::StateManager(uint32_t)
{
    for (int32_t i = 0; i < (int32_t)max_components_; i++) {
        components_.emplace(i, Optional<TypeInfo>::none());
    }

    for (int32_t i = 0; i < (int32_t)max_archetypes_; i++) {
        archetypes_.emplace(i, Optional<ArchetypeStore>::none());
    }

    registerComponent<Entity>();
    registerComponent<WorldID>();
}

void StateManager::registerComponent(uint32_t id, uint32_t alignment,
                                     uint32_t num_bytes)
{
    components_[id].emplace(TypeInfo {
        .alignment = alignment,
        .numBytes = num_bytes,
    });
}

StateManager::ArchetypeStore::ArchetypeStore(uint32_t offset,
                                             uint32_t num_user_components,
                                             uint32_t num_columns,
                                             TypeInfo *type_infos,
                                             IntegerMapPair *lookup_input)
    : componentOffset(offset),
      numUserComponents(num_user_components),
      tbl {},
      columnLookup(lookup_input, num_user_components)
{
    tbl.numRows = 0;
    for (int i = 0 ; i < (int)num_columns; i++) {
        tbl.columns[i] = malloc(type_infos[i].numBytes * maxRowsPerTable);
    }
}

void StateManager::registerArchetype(uint32_t id, ComponentID *components,
                                     uint32_t num_user_components)
{
    uint32_t offset = archetype_component_offset_;
    archetype_component_offset_ += num_user_components;

    uint32_t num_total_components = num_user_components + 2;

    std::array<TypeInfo, max_archetype_components_> type_infos;
    std::array<IntegerMapPair, max_archetype_components_> lookup_input;

    TypeInfo *type_ptr = type_infos.data();

    // Add entity column as first column of every table
    *type_ptr = *components_[0];
    type_ptr++;

    // Add world ID column as second column of every table
    *type_ptr = *components_[1];
    type_ptr++;

    for (int i = 0; i < (int)num_user_components; i++) {
        ComponentID component_id = components[i];
        archetype_components_[offset + i] = component_id.id;

        type_ptr[i] = *components_[component_id.id];

        lookup_input[i] = IntegerMapPair {
            .key = component_id.id,
            .value = (uint32_t)i + user_component_offset_,
        };
    }

    archetypes_[id].emplace(offset, num_user_components,
                            num_total_components,
                            type_infos.data(),
                            lookup_input.data());
}

void StateManager::makeQuery(const uint32_t *components,
                             uint32_t num_components,
                             QueryRef *query_ref)
{
    query_data_lock_.lock();

    if (query_ref->numMatchingArchetypes != 0xFFFF'FFFF) {
        return;
    }

    uint32_t query_offset = query_data_offset_;

    uint32_t num_matching_archetypes = 0;
    for (int32_t archetype_idx = 0; archetype_idx < (int32_t)num_archetypes_;
         archetype_idx++) {
        auto &archetype = *archetypes_[archetype_idx];

        bool has_components = true;
        for (int component_idx = 0; component_idx < (int)num_components; 
             component_idx++) {
            uint32_t component = components[component_idx];
            if (component == TypeTracker::typeID<Entity>()) {
                continue;
            }

            if (!archetype.columnLookup.exists(component)) {
                has_components = false;
                break;
            }
        }

        if (!has_components) {
            continue;
        }

        num_matching_archetypes += 1;
        query_data_[query_data_offset_++] = uint32_t(archetype_idx);

        for (int32_t component_idx = 0;
             component_idx < (int32_t)num_components; component_idx++) {
            uint32_t component = components[component_idx];
            assert(component != TypeTracker::unassignedTypeID);
            if (component == TypeTracker::typeID<Entity>()) {
                query_data_[query_data_offset_++] = 0;
            } else if (component == TypeTracker::typeID<WorldID>()) {
                query_data_[query_data_offset_++] = 1;
            } else {
                query_data_[query_data_offset_++] = 
                    archetype.columnLookup[component];
            }
        }
    }

    query_ref->offset = query_offset;
    query_ref->numMatchingArchetypes = num_matching_archetypes;
    query_ref->numComponents = num_components;

    query_data_lock_.unlock();
}

}
