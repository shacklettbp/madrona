#include <mutex>
#include <memory>
#include <thread>
#include <madrona/net/net.hpp>
#include <madrona/dyn_array.hpp>

#include "comm.hpp"

namespace madrona::net {

struct CheckpointClient::Impl {
    Socket sock;

    // A trajectory can be made of multiple trajectory snapshots
    std::unordered_map<TrajectoryID, Trajectory> trajectories;

    uint8_t *recvBuf;

    std::mutex mu;
    std::thread recvThread;

    Impl(const Config &cfg);
};

CheckpointClient::CheckpointClient(const Config &cfg)
    : impl_(new Impl(cfg))
{
}

CheckpointClient::CheckpointClient(CheckpointClient &&) = default;

CheckpointClient::~CheckpointClient() = default;

CheckpointClient::Impl::Impl(const Config &cfg)
    : sock(Socket::make(Socket::Type::Stream)),
      trajectories(64),
      recvBuf((uint8_t *)malloc(1024 * 64))
{
    if (!sock.connectTo(cfg.ipv4, cfg.port)) {
        FATAL("Failed to connect to server %s, port %d\n",
                cfg.ipv4, (uint32_t)cfg.port);
    }

    // We will be using non-blocking sockets for now
    sock.setBlockingMode(false);
}

void CheckpointClient::requestTrajectory(
        uint32_t world_id, uint32_t num_steps)
{
    uint8_t buf[128];

    DataSerial ds = {
        .ptr = buf,
        .size = 128,
        .offset = 0
    };

    ds.write((uint8_t)PacketType::TrajectoryRequest);
    ds.write(world_id);
    ds.write(num_steps);

    impl_->sock.send((const char *)buf, ds.offset);
}

void CheckpointClient::update()
{
    { // Procedure for receiving a single packet.
        PacketType type;
        if (impl_->sock.receive(&type, sizeof(type)) != sizeof(type)) {
            // Nothing was received from the server.
            return;
        }

        switch (type) {
        case PacketType::ServerShutdown: {
            // TODO: Handle
        } break;

        case PacketType::TrajectoryPackage: {
            uint32_t packet_size;
            assert(impl_->sock.receive(
                    &packet_size, sizeof(packet_size)) == sizeof(type));

            void *data = malloc(packet_size);
            assert(impl_->sock.receive(
                    data, packet_size) == packet_size);

            DataSerial ds = {
                .ptr = (uint8_t *)data,
                .size = packet_size,
                .offset = 0,
            };

            // The trajectory ID contains the world ID in it.
            TrajectoryID traj_id = ds.read<uint32_t>();

            TrajectorySnapshot snapshot = {
                .ckptData = data,
            };

            if (impl_->trajectories.find(traj_id) == impl_->trajectories.end()) {
                impl_->trajectories.emplace(std::make_pair(traj_id, Trajectory { "", {} }));
            }

            impl_->trajectories[traj_id].snapshots.push_back(snapshot);
        } break;

        default: {
            FATAL("Client received invalid packet\n");
        } break;
        }
    }

}

}
