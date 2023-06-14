#pragma once

#include <madrona/macros.hpp>

#if defined(MADRONA_LINUX) || defined(MADRONA_WINDOWS)

#include <madrona/render/vk/backend.hpp>
#include <madrona/render/vk/shader.hpp>

namespace madrona::render {
namespace backend = vk;
}

#elif defined(MADRONA_APPLE)

#include <madrona/render/metal/backend.hpp>

namespace madrona::render {
namespace backend = metal;
}

#endif
