add_library(madrona_net STATIC
    socket.cpp
    client.cpp
    server.cpp
)

target_link_libraries(madrona_net PRIVATE
    madrona_common
)

if (TARGET madrona_cuda)
    target_link_libraries(madrona_net PRIVATE
        madrona_cuda
    )
endif()
