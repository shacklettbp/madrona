#include <queue>
#include <madrona/crash.hpp>
#include <madrona/net/net.hpp>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/cuda_utils.hpp>
#endif

#include "comm.hpp"

namespace madrona::net {

struct PendingRequest {
    enum class RequestType {
        TrajectoryRequest
        // ...
    };

    RequestType type;

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
};

struct CheckpointServer::Impl {
    Socket acceptSock;
    std::vector<Client> clients;

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
{
    // We will be using non-blocking sockets for now
    acceptSock.setBlockingMode(false);
}

void CheckpointServer::Impl::acceptConnections()
{
    for (;;) {
        auto [sock, addr] = acceptSock.acceptConnection();
        if (sock.hdl == -1) {
            return;
        }

        // We have a new client
        Client cl = {
            .sock = sock,
            .addr = addr,
            .numPendingRequests = 0,
        };

        clients.push_back(cl);
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

        { // Queue up new requests
            PacketType type;
            if (cl.sock.receive(&type, sizeof(type)) != sizeof(type)) {
                continue;
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
                };

                req.trajRequest = {
                    .worldID = world_id,
                    .numSteps = num_steps,
                };

                cl.pendingRequests.push(req);
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
    }

    if (processing_requests.size()) {
        uint32_t total_num_ckpts = 0;
        for (uint32_t i = 0; i < processing_requests.size(); ++i) {
            total_num_ckpts += processing_requests[i].trajRequest.numSteps;
        }

        struct TrajInfo {
            uint32_t worldID;
            uint32_t numSteps;
            uint32_t offset;
        };

        // Prepare to request information from the GPU
        size_t readback_bytes = 
            sizeof(uint32_t) +
            // GPU will write 0 or 1 to these uint32_t saying whether
            // there is enough data to send to the client.
            sizeof(uint32_t) * processing_requests.size() +
            // GPU will read from here to see what world / num steps / prefix sum
            // requested is.
            sizeof(TrajInfo) * processing_requests.size() +
            // Sizes of the checkpoint data.
            sizeof(uint32_t) * total_num_ckpts +
            // Pointers to the requested checkpoint data.
            sizeof(void *) * total_num_ckpts;

#ifdef MADRONA_CUDA_SUPPORT
        uint32_t *readback_ptr = (uint32_t *)cu::allocReadback(readback_bytes);
#else
        uint32_t *readback_ptr = (uint32_t *)malloc(1);
        FATAL("Cannot run checkpoint server on non CUDA machine\n");
#endif
        readback_ptr[0] = total_num_ckpts;

        uint32_t *traj_avails = readback_ptr + 1;
        TrajInfo *traj_infos = (TrajInfo *)(
                traj_avails + processing_requests.size());
        uint32_t *ckpt_sizes = (uint32_t *)(
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
    }
}

void CheckpointServer::Impl::update(
    void (*query_ckpt_fn)(void *fn_data, uint32_t num_requests, void *params),
    void *fn_data)
{
    acceptConnections();
    handleRequests(query_ckpt_fn, fn_data);
}

void CheckpointServer::updateImpl(
    void (*query_ckpt_fn)(void *fn_data, uint32_t num_requests, void *params),
    void *fn_data)
{
    impl_->update(query_ckpt_fn, fn_data);
}

}
