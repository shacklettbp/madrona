#pragma once

#include <madrona/sync.hpp>

#include <cstdint>

namespace madrona {

class TypeTracker {
public:
    template <typename T>
    static inline uint32_t typeID()
    {
        return TypeID<T>::id;
    }

    template <typename T>
    static inline uint32_t registerType(uint32_t *next_id_ptr)
    {
        register_lock_.lock();

        if (TypeID<T>::id == 0xFFFF'FFFF) {
            uint32_t next_id = (*next_id_ptr)++;
            TypeID<T>::id = next_id;
        }

        register_lock_.unlock();

        return TypeID<T>::id;
    }

private:
    template <typename T>
    struct TypeID {
        static inline uint32_t id = 0xFFFF'FFFF;
    };

    static inline utils::SpinLock register_lock_ {};

};

}
