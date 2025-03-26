add_library(madrona_net STATIC
    socket.cpp
)

target_link_libraries(madrona_net PRIVATE
    madrona_common
)
