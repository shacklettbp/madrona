#include <madrona/system.hpp>

namespace madrona {

SystemBase::SystemBase(EntryFn fn_ptr)
    : numInvocations(0),
      entry_fn_(fn_ptr)
{}

}
