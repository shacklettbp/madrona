/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include <madrona/type_tracker.hpp>
#include <madrona/memory.hpp>
#include <madrona/utils.hpp>
#include <madrona/dyn_array.hpp>
#include <madrona/sync.hpp>

#include <string_view>
#include <mutex>

namespace madrona {

struct IDInfo {
    uint32_t *ptr;
    std::string_view typeName;
    std::size_t nameHash;
};

struct TrackerImpl {
    SpinLock typeLock;
    DynArray<IDInfo, InitAlloc> ids;
};

static TrackerImpl & getSingletonImpl()
{
    static TrackerImpl impl {
        .typeLock {},
        .ids { 0, InitAlloc() },
    };

    return impl;
}

static std::string_view extractTypeName(const char *compiler_name)
{
    static const char compiler_prefix[] =
#if defined(__clang__)
        "static uint32_t madrona::TypeTracker::trackType(uint32_t *) [T = "
#elif defined(__GNUC__)
        "static uint32_t madrona::TypeTracker::trackType(uint32_t*) [with T = "
#elif defined(_MSC_VER)
        "static uint32_t madrona::TypeTracker::trackType<"
#else 
        STATIC_UNIMPLEMENTED()
#endif
        ;

    compiler_name += sizeof(compiler_prefix) - 1;

    std::string_view type_name(compiler_name);

    size_t end_pos =
#if defined(__clang__)
        type_name.find_last_of(']')
#elif defined(__GNUC__)
        type_name.find_last_of(';')
#elif defined(_MSC_VER)
        type_name.find_last_of('>')
#else 
        STATIC_UNIMPLEMENTED()
#endif
        ;

    return type_name.substr(0, end_pos);
}

// This function is called during static initialization of the TypeID<T>::id
// member for all component / archetypes that are odr-used in a module.
// It doesn't actually perform the type -> id mapping; the return type is
// always ~0u, except when the type has already been registered in
// user code. To allow assigning the ID later, this code stores the pointer
// to the global ID variable for the type, as well as the compiler name.
// For a given type, there can be multiple entries if the type is used across
// library boundaries.
//
// Once TypeTracker::registerType is called later, the tracked ID pointers for
// the type collected during program initialization will all be updated with
// the assigned ID. This ensures that the IDs assigned to each type are
// strictly defined by registration order, rather than implementation defined
// global initialization order.
uint32_t TypeTracker::trackByName(uint32_t *ptr, const char *compiler_name)
{
    std::string_view type_name = extractTypeName(compiler_name);
    size_t type_hash = std::hash<std::string_view>{}(type_name);

    TrackerImpl &tracker = getSingletonImpl();

    std::lock_guard lock(tracker.typeLock);

    uint32_t cur_type_id = unassignedTypeID;

    // This loop ensures that a shared library loaded after type A has already
    // been registered will get the final ID assigned to type A. Otherwise,
    // since registerType won't be called again for that type, the shared
    // library's ID variable would be left at Entity::none() even though
    // it is tracked in the typeInfos list.
    for (const IDInfo &id_info : tracker.ids) {
        if (type_hash == id_info.nameHash &&
            type_name == id_info.typeName) {
            cur_type_id = *id_info.ptr;
            break;
        }
    }

    tracker.ids.push_back({
        .ptr = ptr,
        .typeName = type_name,
        .nameHash = type_hash,
    });

    // This is a weird situation where if multiple threads were calling
    // into trackByName, the lock is released before the code that 
    // calls this function takes the return value and stores it in the
    // global ID variable. This can cause a race with the above for loop,
    // where type_info.ptr still points to uninitialized memory. To avoid
    // this, assign the entity value (usually ~0u) here, even
    // though it will be immediately assigned afterwards as well.
    *ptr = cur_type_id;
    return cur_type_id;
}

// Actually assign the type IDs, this is called by user code that explicitly
// registers components and archetypes.
//
// Performance isn't a huge issue here, since registration should only happen
// in user initialization code, but this could be optimized by switching
// to some kind of hash map on names, at the cost of memory usage.
void TypeTracker::registerType(uint32_t *ptr, uint32_t *next_id_ptr)
{
    auto &tracker = getSingletonImpl();

    std::lock_guard lock(tracker.typeLock);

    // Already registered, presumably multiple StateManagers in use
    if (*ptr != unassignedTypeID) {
        return;
    }

    uint32_t type_id = (*next_id_ptr)++;

    const IDInfo *matched_id_info = nullptr;
    for (const IDInfo &candidate_id_info : tracker.ids) {
        if (candidate_id_info.ptr == ptr) {
            matched_id_info = &candidate_id_info;
            break;
        }
    }

    // If type_info is null, that implies the static member initializer
    // for this type was never called, which makes no sense.
    assert(matched_id_info != nullptr);

    // Find all registered ID memory locations that refer to this type and
    // set them with the newly assigned ID. This covers all code
    // (libraries etc) that is loaded before the call to registerType.
    for (const IDInfo &id_info : tracker.ids) {
        if (matched_id_info->nameHash == id_info.nameHash &&
            matched_id_info->typeName == id_info.typeName) {
            *id_info.ptr = type_id;
        }
    }
}

}
