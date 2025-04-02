#include <madrona/crash.hpp>
#include <madrona/net/net.hpp>

#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
// ...
#else
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netdb.h>
#include <fcntl.h>
#include <errno.h>
#include <netinet/in.h>
#endif

namespace madrona::net {

Socket Socket::make(Socket::Type type)
{
    switch (type) {
    case Type::Stream: {
        return {
            socket(AF_INET, SOCK_STREAM, IPPROTO_TCP),
            type,
        };
    } break;

    case Type::Datagram: {
        return {
            socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP),
            type,
        };
    } break;

    default: {
        FATAL("Specified invalid socket type\n");
        return {};
    } break;
    }
};

void Socket::setRecvBufferSize(uint32_t size)
{
    if (setsockopt(hdl, SOL_SOCKET, SO_RCVBUF, &size, sizeof(size)) == -1) {
        FATAL("Failed to set socket recv buffer size\n");
    }
}

void Socket::setSendBufferSize(uint32_t size)
{
    if (setsockopt(hdl, SOL_SOCKET, SO_SNDBUF, &size, sizeof(size)) == -1) {
        FATAL("Failed to set socket send buffer size\n");
    }
}

uint16_t Socket::bindToPort(Address addr)
{
    sockaddr_in addr_struct = {};
    addr_struct.sin_family = AF_INET;
    addr_struct.sin_port = htons(addr.port);
    addr_struct.sin_addr.s_addr = INADDR_ANY;

    if (bind(hdl, (sockaddr *)&addr_struct, sizeof(addr_struct)) < 0) {
        printf("Failed to bind to port %u\n", (uint32_t)addr.port);
        return 0xFFFF;
    } else {
        return ntohs(addr.port);
    }
}

std::pair<Socket, Address> Socket::acceptConnection()
{
    sockaddr_in from_addr = {};
    socklen_t from_size = sizeof(from_addr);

    int new_hdl = accept(hdl, (sockaddr *)&from_addr, &from_size);

    return {
        Socket { new_hdl, type },
        Address { from_addr.sin_port, from_addr.sin_addr.s_addr },
    };
}

void Socket::setToListening(uint32_t max_clients)
{
    if (listen(hdl, max_clients) < 0) {
        FATAL("Failed to set socket to listening\n");
    }
}

void Socket::setBlockingMode(bool blocking)
{
    int32_t flags = fcntl(hdl, F_GETFL, 0);

    if (flags == -1) {
        // Error
    }

    if (blocking) {
        // Enable blocking
        flags &= ~O_NONBLOCK;
    } else {
        // Enable non-blocking
        flags |= O_NONBLOCK;
    }

    fcntl(hdl, F_SETFL, flags);
}

uint32_t Socket::receiveImpl(char *buf, uint32_t buf_size)
{
    int32_t bytes_received = recv(hdl, buf, buf_size, 0);

    if (bytes_received < 0) {
        return 0;
    }

    return bytes_received;
}

bool Socket::send(const char *buf, uint32_t buf_size)
{
    int32_t send_ret = ::send(hdl, buf, buf_size, MSG_NOSIGNAL);

    if (send_ret < 0) {
        return false;
    } else {
        return true;
    }
}

bool Socket::connectTo(const char *addr_name, uint16_t port)
{
    addrinfo hints = {}, *addresses;

    hints.ai_family = AF_INET;

    switch (type) {
    case Type::Datagram: {
        hints.ai_socktype = SOCK_DGRAM;
        hints.ai_protocol = IPPROTO_UDP;
        break;
    }
        
    case Type::Stream: {
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_protocol = IPPROTO_TCP;
        break;
    }
    }

    char port_str[16] = {};
    sprintf(port_str, "%d", port);

    int32_t err = getaddrinfo(addr_name, port_str, &hints, &addresses);

    if (err != 0) {
        fprintf(stderr, "%s: %s\n", addr_name, gai_strerror(err));
    }

    // (This is a really bad way)
    // Find way to get correct address
    if (addresses) {
        for (addrinfo *addr = addresses; addr != NULL; addr = addr->ai_next) {
            if (addr->ai_family == AF_INET) {
                err = connect(hdl, addr->ai_addr, addr->ai_addrlen);

                freeaddrinfo(addresses);
                if (err < 0) {
                    printf("Failed to connect to address with connect() with %d (%s)\n",
                            errno, strerror(errno));

                    return false;
                }
                else {
                    return true;
                }
            }
        }

        printf("Couldn't find address %s\n", addr_name);

        return false;
    }

    return false;
}

void init()
{
#ifdef _WIN32
    // TODO: Implement WinSock interface.
    FATAL("Sockets interface wasn't implemented for Windows\n");
#endif
}

}
