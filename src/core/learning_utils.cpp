#include <madrona/learning_utils.hpp>

namespace madrona {

namespace LearningMetadataSystem {

void registerTypes(ECSRegistry &registry)
{
    registry.registerStateLogEntry<EpisodeFinished>();
}

}

}
