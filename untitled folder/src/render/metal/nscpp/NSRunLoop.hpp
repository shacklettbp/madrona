#pragma once

#include <Foundation/Foundation.hpp>

namespace NS {

using RunLoopMode = String *;

}

extern "C" NS::RunLoopMode NSDefaultRunLoopMode;

namespace NS {

_NS_INLINE RunLoopMode DefaultRunLoopMode()
{
    return NSDefaultRunLoopMode;
}

}
