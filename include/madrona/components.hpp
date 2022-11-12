#pragma once
#include <madrona/math.hpp>

namespace madrona {
namespace phys {

struct Position : math::Vector3 {
    Position(math::Vector3 v)
        : Vector3(v)
    {}
};

struct Rotation : madrona::math::Quat {
    Rotation(math::Quat q)
        : Quat(q)
    {}
};

}
}
