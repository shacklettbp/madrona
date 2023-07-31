set(INC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../include/madrona")

set(MADRONA_PHYSICS_SRCS
    ${INC_DIR}/physics.hpp ${INC_DIR}/physics.inl physics.cpp
    ${INC_DIR}/mesh_bvh.hpp ${INC_DIR}/mesh_bvh.inl
    narrowphase.cpp broadphase.cpp
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

add_library(madrona_physics_assets STATIC
    ${INC_DIR}/physics_assets.hpp physics_assets.cpp
    mesh_bvh_builder.cpp
)

target_link_libraries(madrona_physics_assets PRIVATE
    madrona_common
)

add_library(madrona_physics_loader STATIC
    ${INC_DIR}/physics_loader.hpp physics_loader.cpp
)

target_link_libraries(madrona_physics_loader
    PRIVATE
        madrona_common
    PUBLIC
        madrona_physics_assets
)

if (TARGET madrona_cuda)
    target_link_libraries(madrona_physics_loader PRIVATE
        madrona_cuda)
endif ()
