#pragma once

#include <madrona/registry.hpp>

namespace madrona {

struct EpisodeFinished {};

namespace LearningMetadataSystem {

void registerTypes(ECSRegistry &registry);

}

}
