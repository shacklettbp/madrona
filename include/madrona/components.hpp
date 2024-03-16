#pragma once
#include <madrona/math.hpp>
#include <madrona/fwd.hpp>
#include <madrona/taskgraph.hpp>

namespace madrona {
namespace base {

struct Position : math::Vector3 {
    Position(math::Vector3 v)
        : Vector3(v)
    {}
};

struct Rotation : math::Quat {
    Rotation(math::Quat q)
        : Quat(q)
    {}
};

struct Scale : math::Diag3x3 {
    Scale(math::Diag3x3 d)
        : Diag3x3(d)
    {}
};

struct ObjectID {
    int32_t idx;
};

struct ObjectInstance : Bundle<
    Position, 
    Rotation,
    Scale,
    ObjectID
> {};

void registerTypes(ECSRegistry &registry);

}
}
