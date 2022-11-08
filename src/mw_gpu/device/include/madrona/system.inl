#pragma once

namespace madrona {
namespace mwGPU {

template <typename SystemT>
__attribute__((used, always_inline))
inline void systemEntry(SystemBase *sys_base, void *user_data,
                        uint32_t invocation_offset)
{
    SystemT::entry(sys_base, user_data, invocation_offset);
}

template <typename SystemT>
struct SystemIDBase {
    static uint32_t id;
};

template <typename SystemT,
          decltype(systemEntry<SystemT>) =
              systemEntry<SystemT>>
struct SystemID : SystemIDBase<SystemT> {};

}

SystemBase::SystemBase(uint32_t sys_id)
    : sys_id_(sys_id)
{}

template <typename SystemT>
CustomSystem<SystemT>::CustomSystem()
    : SystemBase(mwGPU::SystemID<SystemT>::id)
{}

template <typename SystemT>
void CustomSystem<SystemT>::entry(SystemBase *sys_base, void *data,
                                  uint32_t invocation_offset)
{
    SystemT *sys = static_cast<SystemT *>(sys_base);
    sys->run(data, invocation_offset);
}

}
