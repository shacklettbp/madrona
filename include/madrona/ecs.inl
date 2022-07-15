#pragma once

namespace madrona {

constexpr Entity Entity::none()
{
    return Entity {
        ~0u,
    };
}

bool operator==(Entity a, Entity b)
{
    return a.id == b.id;
}

bool operator!=(Entity a, Entity b)
{
    return !(a == b);
}

}
