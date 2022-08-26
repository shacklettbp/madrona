#pragma once

namespace madrona {

constexpr Entity Entity::none()
{
    return Entity {
        ~0u,
        ~0u,
        ~0u,
    };
}

bool Loc::valid() const
{
    return idx != ~0u;
}

bool operator==(Entity a, Entity b)
{
    return a.gen == b.gen && a.archetype == b.archetype &&
        a.id == b.id;
}

bool operator!=(Entity a, Entity b)
{
    return !(a == b);
}

bool operator==(Loc a, Loc b)
{
    return a.idx == b.idx;
}

bool operator!=(Loc a, Loc b)
{
    return !(a == b);
}


}
