#pragma once

namespace madrona {

constexpr Entity Entity::none()
{
    return Entity {
        ~0u,
        ~0u,
    };
}

bool Loc::valid() const
{
    return archetype != ~0u;
}

Loc Loc::none()
{
    return Loc {
        .archetype = ~0u,
        .row = 0,
    };
}

bool operator==(Entity a, Entity b)
{
    return a.gen == b.gen && a.id == b.id;
}

bool operator!=(Entity a, Entity b)
{
    return !(a == b);
}

bool operator==(Loc a, Loc b)
{
    return a.row == b.row && a.archetype == b.archetype;
}

bool operator!=(Loc a, Loc b)
{
    return !(a == b);
}


}
