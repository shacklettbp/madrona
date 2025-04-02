#include <queue>
#include <madrona/crash.hpp>
#include <madrona/net/net.hpp>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/cuda_utils.hpp>
#endif

#include "comm.hpp"

#include <chrono>

namespace madrona::net {

struct PendingRequest {
    enum class RequestType {
        TrajectoryRequest
        // ...
    };

    RequestType type;
    uint32_t clientID;

    union {
        struct {
            uint32_t worldID;
            uint32_t numSteps;
        } trajRequest;

        // ...
    };
};

struct Client {
    Socket sock;
    Address addr;
    uint32_t numPendingRequests;

    std::queue<PendingRequest> pendingRequests;

    std::chrono::time_point<
        std::chrono::system_clock> lastPing;

    bool toDelete;
};

struct CheckpointServer::Impl {
    Socket acceptSock;
    std::vector<Client> clients;

    uint32_t trajectoryCount;

    Impl(const Config &cfg);

    void update(
        void (*query_ckpt_fn)(void *fn_data, uint32_t num_requests, void *params),
        void *fn_data);

    void acceptConnections();
    void handleRequests(
        void (*query_ckpt_fn)(void *fn_data, uint32_t num_requests, void *params),
        void *fn_data);
};

CheckpointServer::Impl::Impl(const Config &cfg)
    : acceptSock(Socket::make(Socket::Type::Stream)),
      trajectoryCount(0)
{
    // We will be using non-blocking sockets for now
    acceptSock.setBlockingMode(false);

    acceptSock.bindToPort({ cfg.bindPort });
    acceptSock.setToListening(cfg.maxClients);
}

void CheckpointServer::Impl::acceptConnections()
{
    for (;;) {
        auto [sock, addr] = acceptSock.acceptConnection();
        if (sock.hdl == -1) {
            return;
        }

        sock.setBlockingMode(false);

        // We have a new client
        Client cl = {
            .sock = sock,
            .addr = addr,
            .numPendingRequests = 0,
            .lastPing = std::chrono::system_clock::now(),
        };

        clients.push_back(cl);
        
        printf("New client just connected!\n");
    }
}

void CheckpointServer::Impl::handleRequests(
    void (*query_ckpt_fn)(void *fn_data, uint32_t num_requests, void *params),
    void *fn_data)
{
    // These are requests which are currently processing.
    // We need to do some work on the GPU to get some of the information
    // we want to know before we can send off information to clients.
    std::vector<PendingRequest> processing_requests;

    for (uint32_t i = 0; i < clients.size(); ++i) {
        Client &cl = clients[i];

        for (;;) { // Queue up new requests
            PacketType type;
            if (cl.sock.receive(&type, sizeof(type)) != sizeof(type)) {
                break;
            }

            switch (type) {
            case PacketType::TrajectoryRequest: {
                uint32_t world_id, num_steps;
                
                assert(cl.sock.receive(
                        &world_id, sizeof(world_id)) == sizeof(world_id));
                assert(cl.sock.receive(
                        &num_steps, sizeof(num_steps)) == sizeof(num_steps));

                PendingRequest req = {
                    .type = PendingRequest::RequestType::TrajectoryRequest,
                    .clientID = i,
                };

                req.trajRequest = {
                    .worldID = world_id,
                    .numSteps = num_steps,
                };

                cl.pendingRequests.push(req);

                printf("Received trajectory request!\n");
            } break;

            case PacketType::Ping: {
                cl.lastPing = std::chrono::system_clock::now();
            } break;

            default: {
                FATAL("Server received invalid packet\n");
            } break;
            }
        }

        if (!cl.pendingRequests.empty()) { // Process current requests
            PendingRequest req = cl.pendingRequests.front();
            processing_requests.push_back(req);
        }

        { // Check if client isn't responding anymore
            auto now = std::chrono::system_clock::now();
            float dt = std::chrono::duration_cast<std::chrono::seconds>(now - cl.lastPing).count();

            if (dt > 1.f) {
                printf("Didn't receive client ping in a while; disconnecting\n");
                cl.toDelete = true;
            }
        }
    }

    if (processing_requests.size()) {
        uint32_t total_num_ckpts = 0;
        for (uint32_t i = 0; i < processing_requests.size(); ++i) {
            total_num_ckpts += processing_requests[i].trajRequest.numSteps;
        }

        struct TrajInfo {
            uint64_t worldID;
            uint64_t numSteps;
            uint64_t offset;
        };

        // Prepare to request information from the GPU
        size_t readback_bytes = 
            sizeof(uint64_t) +
            // GPU will write 0 or 1 to these uint32_t saying whether
            // there is enough data to send to the client.
            sizeof(uint64_t) * processing_requests.size() +
            // GPU will read from here to see what world / num steps / prefix sum
            // requested is.
            sizeof(TrajInfo) * processing_requests.size() +
            // Sizes of the checkpoint data.
            sizeof(uint64_t) * total_num_ckpts +
            // Pointers to the requested checkpoint data.
            sizeof(void *) * total_num_ckpts;

#ifdef MADRONA_CUDA_SUPPORT
        uint64_t *readback_ptr = (uint64_t *)cu::allocReadback(readback_bytes);
#else
        uint64_t *readback_ptr = (uint64_t *)malloc(1);
        FATAL("Cannot run checkpoint server on non CUDA machine\n");
#endif
        readback_ptr[0] = total_num_ckpts;

        uint64_t *traj_avails = readback_ptr + 1;
        TrajInfo *traj_infos = (TrajInfo *)(
                traj_avails + processing_requests.size());
        uint64_t *ckpt_sizes = (uint64_t *)(
                traj_infos + processing_requests.size());
        void **ckpt_ptrs = (void **)(ckpt_sizes + total_num_ckpts);

        // Calculate prefix sum.
        uint32_t offset = 0;
        for (uint32_t i = 0; i < processing_requests.size(); ++i) {
            traj_infos[i] = {
                .worldID = processing_requests[i].trajRequest.worldID,
                .numSteps = processing_requests[i].trajRequest.numSteps,
                .offset = offset,
            };

            offset += processing_requests[i].trajRequest.numSteps;
        }

        // This should fill in everything in readback_ptr
        query_ckpt_fn(fn_data, processing_requests.size(), readback_ptr);

        for (uint32_t i = 0; i < processing_requests.size(); ++i) {
            if (traj_avails[i] == 1) {
                // Package the trajectory and send it to the client
                uint64_t *traj_ckpt_sizes = ckpt_sizes + traj_infos[i].offset;
                void **traj_ckpt_ptrs = ckpt_ptrs + traj_infos[i].offset;

                // This first uint32_t is for the trajectory ID
                uint64_t data_size = sizeof(TrajectoryID) +
                                     sizeof(uint32_t) + // number of ckpts
                                     sizeof(uint32_t) * traj_infos[i].numSteps +
                                     sizeof(uint32_t) * traj_infos[i].numSteps;
                for (uint32_t c = 0; c < traj_infos[i].numSteps; ++c) {
                    data_size += traj_ckpt_sizes[c];
                }

                uint8_t *packet_data = (uint8_t *)malloc(
                        sizeof(PacketType) + sizeof(uint32_t) + data_size);

                DataSerial ds = {
                    .ptr = packet_data,
                    .size = (uint32_t)(sizeof(PacketType) + // Packet type
                                       sizeof(uint32_t) +   // Packet size
                                       data_size),
                    .offset = 0,
                };

                ds.write(PacketType::TrajectoryPackage);
                ds.write((uint32_t)data_size);

                printf("Sending packet with size %u\n", (uint32_t)data_size);

                TrajectoryID traj_id = ((uint16_t)trajectoryCount << 16) |
                                        (uint16_t)traj_infos[i].worldID;

                // Start of actual data
                ds.write(traj_id); // Trajectory ID
                ds.write((uint32_t)traj_infos[i].numSteps); // Number of checkpoints

                // Write the checkpoints offsets
                uint64_t traj_offset = 0;
                for (uint32_t c = 0; c < traj_infos[i].numSteps; ++c) {
                    ds.write((uint32_t)traj_offset);
                    traj_offset += traj_ckpt_sizes[c];
                }

                // Write the checkpoint sizes
                for (uint32_t c = 0; c < traj_infos[i].numSteps; ++c) {
                    ds.write((uint32_t)traj_ckpt_sizes[c]);
                }

                // Now, write all the checkpoint data
                for (uint32_t c = 0; c < traj_infos[i].numSteps; ++c) {
                    uint64_t num_bytes = traj_ckpt_sizes[c];
                    void *gpu_ptr = traj_ckpt_ptrs[c];
                    ds.writeFromGPU(gpu_ptr, (uint32_t)num_bytes);
                }

                // Send to client
                Client &cl = clients[processing_requests[i].clientID];
                cl.sock.send((const char *)ds.ptr, ds.size);

                // cl.pendingRequests.pop_front();
                cl.pendingRequests.pop();

                trajectoryCount++;
            }
        }
    }

    uint32_t num_clients_pre = clients.size();
    clients.erase(std::remove_if(clients.begin(), clients.end(),
        [&](Client &cl) mutable { return cl.toDelete; }),
        clients.end());

    uint32_t num_clients_post = clients.size();
    if (num_clients_post < num_clients_pre) {
        printf("Deleted client\n");
    }
}

void CheckpointServer::Impl::update(
    void (*query_ckpt_fn)(void *fn_data, uint32_t num_requests, void *params),
    void *fn_data)
{
    handleRequests(query_ckpt_fn, fn_data);
    acceptConnections();
}

void CheckpointServer::updateImpl(
    void (*query_ckpt_fn)(void *fn_data, uint32_t num_requests, void *params),
    void *fn_data)
{
    impl_->update(query_ckpt_fn, fn_data);
}

CheckpointServer::CheckpointServer(CheckpointServer &&) = default;
CheckpointServer::~CheckpointServer() = default;

CheckpointServer::CheckpointServer(const Config &cfg)
    : impl_(new Impl(cfg))
{
}

uint32_t CheckpointServer::getNumClients()
{
    return impl_->clients.size();
}

}
