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

    inline uint32_t getNumCheckpoints()
    {
        uint32_t total = 0;
        for (uint32_t i = 0; i < snapshots.size(); ++i) {
            total += snapshots[i].getNumCheckpoints();
        }
        return total;
    }

    inline std::pair<uint32_t, void *> getStepData(uint32_t step)
    {
        uint32_t total_ckpts = 0;
        for (uint32_t i = 0; i < snapshots.size(); ++i) {
            uint32_t prev = total_ckpts;

            total_ckpts += snapshots[i].getNumCheckpoints();
            
            if (step < total_ckpts) {
                uint32_t step_offset = step - prev;

                uint32_t data_offset = snapshots[i].getOffsets()[step_offset];
                uint32_t data_size = snapshots[i].getSizes()[step_offset];
                void *data = (void *)(
                        (uint8_t *)snapshots[i].getCheckpointDataBase() +
                        data_offset);

                return { data_size, data };
            }
        }

        return { 0, nullptr };
    }
};

struct CheckpointClient {
    CheckpointClient();
    CheckpointClient(CheckpointClient &&);
    ~CheckpointClient();

    void connect(const char *ipv4, uint16_t port);

    void requestTrajectory(uint32_t world_id, uint32_t num_steps);

    // TODO: Implement this
    void requestCustom(const char *filter);

    void update();

    std::vector<Trajectory> getTrajectories();

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
