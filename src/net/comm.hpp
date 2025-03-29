#pragma once

#include <stdint.h>
#include <cassert>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/cuda_utils.hpp>
#endif

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

    // Ping to make sure clients are still alive
    Ping,
};

struct DataSerial {
    uint8_t *ptr;
    uint32_t size;
    uint32_t offset;

    template <typename T>
    inline void write(T data)
    {
        assert(offset + sizeof(data) <= size);
        memcpy(ptr + offset, &data, sizeof(T));
        offset += sizeof(T);
    }

    template <typename T>
    inline void write(T *data, uint32_t count)
    {
        assert(offset + sizeof(data) * count <= size);
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

#ifdef MADRONA_CUDA_SUPPORT
    inline void writeFromGPU(void *gpu_ptr, uint32_t num_bytes)
    {
        assert(offset + num_bytes <= size);
        REQ_CUDA(cudaMemcpy(
                    (void *)(ptr + offset), 
                    gpu_ptr,
                    num_bytes,
                    cudaMemcpyDeviceToHost));
        offset += num_bytes;
    }
#endif
};

}
