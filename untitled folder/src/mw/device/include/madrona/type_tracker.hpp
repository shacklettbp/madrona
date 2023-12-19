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

        if (TypeID<T>::id == unassignedTypeID) {
            uint32_t next_id = (*next_id_ptr)++;
            TypeID<T>::id = next_id;
        }

        register_lock_.unlock();

        return TypeID<T>::id;
    }

    static constexpr uint32_t unassignedTypeID = 0xFFFF'FFFF;
private:
    template <typename T>
    struct TypeID {
        static inline uint32_t id = unassignedTypeID;
    };

    static inline SpinLock register_lock_ {};

};

}
