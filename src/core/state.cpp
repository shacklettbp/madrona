#include <madrona/state.hpp>
#include <madrona/utils.hpp>
#include <madrona/dyn_array.hpp>

#include <cassert>

namespace madrona {

struct TypeIDTracker {
    utils::SpinLock typeLock;
    DynArray<const char *, InitAlloc> typeStrings;
};

static TypeIDTracker &getSingletonTracker()
{
    static TypeIDTracker id_tracker {
        .typeLock {},
        .typeStrings { 0, InitAlloc() },
    };

    return id_tracker;
}

uint32_t IDManager::numTypes()
{
    TypeIDTracker &tracker = getSingletonTracker();

    std::lock_guard lock(tracker.typeLock);

    return tracker.typeStrings.size();
}

// Performance doesn't really matter, so to minimize memory
// impact just do a linear search
uint32_t IDManager::assignID(const char *identifier)
{
    TypeIDTracker &tracker = getSingletonTracker();

    std::lock_guard lock(tracker.typeLock);

    for (int i = 0; i < (int)tracker.typeStrings.size(); i++) {
        if (!strcmp(tracker.typeStrings[i], identifier)) {
            return (uint32_t)i;
        }
    }

    tracker.typeStrings.push_back(identifier);

    return tracker.typeStrings.size() - 1;
}

StateManager::StateManager(uint32_t max_components)
{
    assert(max_components >= IDManager::numTypes());

    (void)max_components;
}

}
