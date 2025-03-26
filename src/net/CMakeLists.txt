add_library(madrona_net STATIC
    socket.cpp
    client.cpp
    server.cpp
)

target_link_libraries(madrona_net PRIVATE
    madrona_common
)
