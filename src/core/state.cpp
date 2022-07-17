#include <madrona/state.hpp>
#include <madrona/utils.hpp>
#include <madrona/dyn_array.hpp>

#include <cassert>
#include <functional>
#include <mutex>
#include <string_view>

namespace madrona {

struct StateManager::TypeInfo {
    union {
        struct {
            uint32_t alignment;
            uint32_t numBytes;
        };
        struct {
            uint32_t componentOffset;
            uint32_t numComponents;
        };
    };
};

struct IDInfo {
    Entity *ptr;
    std::string_view typeName;
    std::size_t nameHash;
};

struct TypeIDTracker {
    utils::SpinLock typeLock;
    DynArray<IDInfo, InitAlloc> ids;
    uint32_t numRegisteredTypes;
};

StateManager::StateManager(uint32_t max_components)
    : type_infos_(
        (TypeInfo *)InitAlloc().alloc(max_components * sizeof(TypeInfo))),
      archetype_components_(0)
{}

StateManager::~StateManager()
{
    InitAlloc().dealloc(type_infos_);
}

static TypeIDTracker &getSingletonTracker()
{
    static TypeIDTracker id_tracker {
        .typeLock {},
        .ids { 0, InitAlloc() },
        .numRegisteredTypes = 0,
    };

    return id_tracker;
}

static std::string_view extractTypeName(const char *compiler_name)
{
    static const char compiler_prefix[] =
#ifdef _MSC_VER
        STATIC_UNIMPLEMENTED();
#elif defined(__clang__)
        "static madrona::Entity madrona::StateManager::trackType(madrona::Entity *) [T = "
#elif defined(__GNUC__)
        "static madrona::Entity madrona::StateManager::trackType(madrona::Entity*) [with T = "
#else
        STATIC_UNIMPLEMENTED();
#endif
        ;

    compiler_name += sizeof(compiler_prefix) - 1;

    std::string_view type_name(compiler_name);

    size_t end_pos =
#ifdef _MSC_VER
        STATIC_UNIMPLEMENTED();
#elif defined(__clang__) or defined(__GNUC__)
        type_name.find_last_of(']')
#else
        STATIC_UNIMPLEMENTED();
#endif
        ;

    return type_name.substr(0, end_pos);
}

// This function is called during static initialization of the TypeID<T>::id
// member for all component / archetypes that are odr-used in a module.
// It doesn't actually perform the type -> id mapping; the return type is
// always Entity::none(), except when the type has already been registered in
// user code. To allow assigning the ID later, this code stores the pointer
// to the global ID variable for the type, as well as the compiler name.
// For a given type, there can be multiple entries if the type is used across
// library boundaries.
//
// Once a type has been registered (with StateManager::registerType), the ID
// is assigned based on a global counter, and all the tracked
// ID pointers for the type collected during program initialization are
// updated. This ensures that the IDs assigned to each type are strictly
// defined by registration order, rather than implementation defined global
// initialization order.
Entity StateManager::trackByName(Entity *ptr, const char *compiler_name)
{
    std::string_view type_name = extractTypeName(compiler_name);
    size_t type_hash = std::hash<std::string_view>{}(type_name);

    TypeIDTracker &tracker = getSingletonTracker();

    std::lock_guard lock(tracker.typeLock);

    Entity cur_type_entity = Entity::none();

    // This loop ensures that a shared library loaded after type A has already
    // been registered will get the final ID assigned to type A. Otherwise,
    // since registerType won't be called again for that type, the shared
    // library's ID variable would be left at Entity::none() even though
    // it is tracked in the typeInfos list.
    for (const IDInfo &id_info : tracker.ids) {
        if (type_hash == id_info.nameHash &&
            type_name == id_info.typeName) {
            cur_type_entity = *id_info.ptr;
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
    // this, assign the entity value (usually Entity::none()) here, even
    // though it will be immediately assigned afterwards as well.
    *ptr = cur_type_entity;
    return cur_type_entity;
}

// Actually assign the type IDs, this is called by user code that explicitly
// registers components and archetypes.
//
// Performance isn't a huge issue here, since registration should only happen
// in user initialization code, but this could be optimized by switching
// to some kind of hash map on names, at the cost of memory usage.
void StateManager::registerType(Entity *ptr)
{
    // Already registered, presumably to another StateManager
    if (*ptr != Entity::none()) {
        return;
    }

    auto &tracker = getSingletonTracker();

    std::lock_guard lock(tracker.typeLock);

    uint32_t type_id = tracker.numRegisteredTypes++;

    Entity type_entity {
        .id = type_id,
    };

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
    // (libraries etc) that is loaded before the callto registerType.
    for (const IDInfo &id_info : tracker.ids) {
        if (matched_id_info->nameHash == id_info.nameHash &&
            matched_id_info->typeName == id_info.typeName) {
            *id_info.ptr = type_entity;
        }
    }
}

void StateManager::saveComponentInfo(Entity id, uint32_t alignment,
                                     uint32_t num_bytes)
{
    type_infos_[id.id] = TypeInfo {
        .alignment = alignment,
        .numBytes = num_bytes,
    };
}

void StateManager::saveArchetypeInfo(Entity id, Span<Entity> components)
{
    uint32_t offset = archetype_components_.size();
    for (Entity e : components) {
        archetype_components_.push_back(e);
    }

    type_infos_[id.id] = TypeInfo {
        .componentOffset = offset,
        .numComponents = components.size(),
    };
}


}
