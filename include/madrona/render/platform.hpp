#pragma once

#include <madrona/macros.hpp>

#if defined(MADRONA_LINUX) || defined(MADRONA_WINDOWS)

#include <madrona/render/vk/platform.hpp>

namespace madrona::render {
namespace platform = vk;
}

#elif defined(MADRONA_APPLE)

#include <madrona/render/metal/platform.hpp>

namespace madrona::render {
namespace platform = metal;
}

#endif
