#pragma once

#include <stdint.h>
#include <cassert>

namespace madrona::net {

enum class PacketType : uint8_t {
    // Client sends this
    ClientDisconnect,

    // Server sends this
    ServerShutdown,

    // Client sends this
    TrajectoryRequest,
    // Server sends this
    TrajectoryPackage,

    // Client sends this
    CustomRequest,
};

struct DataSerial {
    uint8_t *ptr;
    uint32_t size;
    uint32_t offset;

    template <typename T>
    inline void write(T data)
    {
        assert(offset < size);
        memcpy(ptr + offset, &data, sizeof(T));
        offset += sizeof(T);
    }

    template <typename T>
    inline void write(T *data, uint32_t count)
    {
        assert(offset < size);
        memcpy(ptr + offset, data, sizeof(T) * count);
        offset += sizeof(T) * count;
    }

    template <typename T>
    inline T read()
    {
        assert(offset < size);
        T data;
        memcpy(&data, ptr + offset, sizeof(T));
        offset += sizeof(T);
        return data;
    }

    template <typename T>
    inline void read(T *dst, uint32_t count)
    {
        assert(offset < size);
        memcpy(dst, ptr + offset, sizeof(T) * count);
        offset += sizeof(T) * count;
    }
};

}
