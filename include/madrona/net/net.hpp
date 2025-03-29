#pragma once

#include <vector>
#include <memory>
#include <string>
#include <utility>
#include <stdint.h>

namespace madrona::net {

// For now, we just support IPv4. TODO: Support IPv6
struct Address {
    uint16_t port;
    uint32_t ipv4Address;
};

using TrajectoryID = uint32_t;

struct Socket {
    enum class Type {
        Stream,
        Datagram
    };

    static Socket make(Type type);
    void setRecvBufferSize(uint32_t size);
    void setSendBufferSize(uint32_t size);

    // Returns 0xFFFF upon failure, and host order port number
    uint16_t bindToPort(Address addr);

    std::pair<Socket, Address> acceptConnection();

    void setToListening(uint32_t max_clients);
    void setBlockingMode(bool blocking);

    template <typename T>
    uint32_t receive(T *buf, uint32_t buf_size)
    {
        return receiveImpl((char *)buf, buf_size);
    }

    bool send(const char *buf, uint32_t buf_size);

    bool connectTo(const char *addr_name, uint16_t port);



    int32_t hdl;
    Type type;

private:
    uint32_t receiveImpl(char *buf, uint32_t buf_size);
};

struct TrajectorySnapshot {
    // This is the pointer returned by malloc
    void *ckptData;

    inline uint32_t getTrajectoryID()
    {
        return *((uint32_t *)ckptData);
    }

    inline uint32_t getNumCheckpoints()
    {
        return *((uint32_t *)ckptData + 1);
    }

    inline uint32_t *getOffsets()
    {
        return (uint32_t *)((uint32_t *)ckptData + 2);
    }

    inline uint32_t *getSizes()
    {
        return (uint32_t *)((uint32_t *)ckptData + 2 + getNumCheckpoints());
    }

    inline void *getCheckpointDataBase()
    {
        return (void *)(getSizes() + getNumCheckpoints());
    }
};

struct Trajectory {
    std::string name;
    std::vector<TrajectorySnapshot> snapshots;
};

struct CheckpointClient {
    struct Config {
        const char *ipv4;
        uint16_t port;
    };

    CheckpointClient(const Config &cfg);
    CheckpointClient(CheckpointClient &&);
    ~CheckpointClient();

    void requestTrajectory(uint32_t world_id, uint32_t num_steps);

    // TODO: Implement this
    void requestCustom(const char *filter);

    void update();

private:
    struct Impl;

    std::unique_ptr<Impl> impl_;
};

struct CheckpointServer {
    struct Config {
        uint32_t maxClients;
        uint16_t bindPort;
    };

    CheckpointServer(const Config &cfg);
    CheckpointServer(CheckpointServer &&);
    ~CheckpointServer();

    template <typename FnT>
    inline void update(FnT &&fn)
    {
        updateImpl([](void *fn_data, uint32_t num_requests, void *params) {
            auto *fn_ptr = (FnT *)fn_data;
            (*fn_ptr)(num_requests, params);
        }, (void *)&fn);
    }

    uint32_t getNumClients();

private:
    struct Impl;

    void updateImpl(
        void (*query_ckpt_fn)(void *fn_data, uint32_t num_requests, void *params),
        void *fn_data);

    std::unique_ptr<Impl> impl_;
};

}
