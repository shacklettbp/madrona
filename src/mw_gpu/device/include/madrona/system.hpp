#pragma once

#include <madrona/fwd.hpp>
#include <madrona/utils.hpp>
#include <madrona/query.hpp>

#include <cstdint>

namespace madrona {

class SystemBase {
public:
    inline SystemBase(uint32_t sys_id);
    std::atomic_uint32_t numInvocations; 

private:
    uint32_t sys_id_;
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
