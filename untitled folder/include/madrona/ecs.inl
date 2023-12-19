/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

namespace madrona {

constexpr Entity Entity::none()
{
    return Entity {
        0xFFFF'FFFF_u32,
        0xFFFF'FFFF_i32,
    };
}

bool Loc::valid() const
{
    return archetype != 0xFFFF'FFFF_u32;
}

Loc Loc::none()
{
    return Loc {
        0xFFFF'FFFF_u32,
        0,
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
