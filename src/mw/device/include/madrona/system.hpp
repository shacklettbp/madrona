#pragma once

#include <madrona/fwd.hpp>
#include <madrona/utils.hpp>
#include <madrona/query.hpp>

#include <cstdint>

namespace madrona {

struct SharedSystemState {
    SystemBase **systems;
    AtomicU32 numInvocations; 
    uint32_t sysID;
};

class SystemBase {
public:
    inline SystemBase(uint32_t sys_id);

private:
    uint32_t sys_id_;
    SharedSystemState * shared_;
friend class TaskGraph;
};

template <typename SystemT>
class CustomSystem : public SystemBase {
public:
    CustomSystem();

    static void entry(SystemBase *sys, void *data,
                      uint32_t invocation_offset);
};

}

#include "system.inl"
