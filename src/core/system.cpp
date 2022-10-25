#include <madrona/system.hpp>

namespace madrona {

SystemBase::SystemBase(EntryFn fn_ptr)
    : fn_ptr_(fn_ptr)
{}

}
