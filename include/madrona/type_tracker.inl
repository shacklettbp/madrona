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
#ifdef MADRONA_MSVC
        __FUNCDNAME__
#else
        __PRETTY_FUNCTION__
#endif
        );
}

}
