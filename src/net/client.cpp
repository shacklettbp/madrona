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

    Impl();
};

CheckpointClient::CheckpointClient()
    : impl_(new Impl())
{
}

CheckpointClient::CheckpointClient(CheckpointClient &&) = default;

CheckpointClient::~CheckpointClient() = default;

CheckpointClient::Impl::Impl()
    : sock(Socket::make(Socket::Type::Stream)),
      trajectories(64),
      recvBuf((uint8_t *)malloc(1024 * 1024))
{
}

void CheckpointClient::connect(const char *ipv4, uint16_t port)
{
    if (!impl_->sock.connectTo(ipv4, port)) {
        FATAL("Failed to connect to server %s, port %d\n",
                ipv4, (uint32_t)port);
    } else {
        printf("Successfully connected to %s:%u\n",
                ipv4, (uint32_t)port);
    }

    // We will be using non-blocking sockets for now
    impl_->sock.setBlockingMode(false);
    impl_->sock.setRecvBufferSize(1024 * 1024);
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

        printf("Received a packet!\n");

        switch (type) {
        case PacketType::ServerShutdown: {
            // TODO: Handle
        } break;

        case PacketType::TrajectoryPackage: {
            uint32_t packet_size;
            assert(impl_->sock.receive(
                    &packet_size, sizeof(packet_size)) == sizeof(packet_size));

            void *data = malloc(packet_size);
            uint32_t read_bytes = 0;

            while (read_bytes < packet_size) {
                int32_t main_recv_bytes = impl_->sock.receive(
                        (void *)((uint8_t *)data + read_bytes), packet_size);

                if (main_recv_bytes < 0) {
                    continue;
                }

                read_bytes += (uint32_t)main_recv_bytes;
                printf("Received %u bytes\n", main_recv_bytes);
            }

            DataSerial ds = {
                .ptr = (uint8_t *)data,
                .size = packet_size,
                .offset = 0,
            };

            // The trajectory ID contains the world ID in it.
            TrajectoryID traj_id = ds.read<uint32_t>();

            printf("Received trajectory with ID %u, packet size %u\n", 
                    traj_id, packet_size);

            TrajectorySnapshot snapshot = {
                .ckptData = data,
            };

            if (impl_->trajectories.find(traj_id) == impl_->trajectories.end()) {
                std::string traj_name = "traj_" + std::to_string(traj_id);
                impl_->trajectories.emplace(std::make_pair(traj_id, 
                            Trajectory { traj_name, {} }));
            }

            impl_->trajectories[traj_id].snapshots.push_back(snapshot);
        } break;

        default: {
            FATAL("Client received invalid packet\n");
        } break;
        }
    }

    { // Send a ping packet
        PacketType type = PacketType::Ping;
        if (impl_->sock.send((const char *)&type, sizeof(type)) != sizeof(type)) {
            printf("Failed to send ping\n");
        }
    }
}

std::vector<Trajectory> CheckpointClient::getTrajectories()
{
    std::vector<Trajectory> trajs;
    for (auto [id, traj] : impl_->trajectories) {
        trajs.push_back(traj);
    }

    return trajs;
}

}
