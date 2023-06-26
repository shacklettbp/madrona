/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

namespace madrona {

template <typename T>
uint32_t TypeTracker::typeID()
{
    return TypeID<T>::id;
}

template <typename T>
void TypeTracker::registerType(uint32_t *next_id_ptr)
{
    uint32_t *ptr = &TypeID<T>::id;

    TypeTracker::registerType(ptr, next_id_ptr);
}

template <typename T>
uint32_t TypeTracker::TypeID<T>::id =
    TypeTracker::trackType<T>(&TypeTracker::TypeID<T>::id);

template <typename T>
uint32_t TypeTracker::trackType(uint32_t *ptr)
{
    return TypeTracker::trackByName(ptr,
#if defined (__clang__) || defined(__GNUC__)
        __PRETTY_FUNCTION__
#elif defined(_MSC_VER)
        __FUNCSIG__
#else
        ""
#endif
    );
}

}
