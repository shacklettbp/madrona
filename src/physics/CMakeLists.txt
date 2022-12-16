set(INC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../include/madrona")

set(MADRONA_PHYSICS_SRCS
    ${INC_DIR}/physics.hpp ${INC_DIR}/physics.inl physics.cpp
    ${INC_DIR}/physics_assets.hpp physics_assets.cpp
)

add_library(madrona_physics STATIC
    ${MADRONA_PHYSICS_SRCS}
)

target_link_libraries(madrona_physics
    PUBLIC
        madrona_core
)

add_library(madrona_mw_physics STATIC
    ${MADRONA_PHYSICS_SRCS}
)

target_link_libraries(madrona_mw_physics
    PUBLIC
        madrona_mw_core
)
