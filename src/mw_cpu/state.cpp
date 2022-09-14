#include <madrona/mw/state.hpp>

namespace madrona {
namespace mw {

StateManager::StateManager(int num_worlds)
    : StateManagerBase(),
      register_lock_(),
      num_worlds_(num_worlds)
{
}

}
}
