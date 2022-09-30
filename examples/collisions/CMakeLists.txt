set(COLLISION_EXAMPLE_SRCS
    collisions.hpp collisions.inl collisions.cpp
)

add_library(collisions_sw_core STATIC
    ${COLLISION_EXAMPLE_SRCS}
)
target_link_libraries(collisions_sw_core PUBLIC
    madrona_core
)

add_library(collisions_mw_core STATIC
    ${COLLISION_EXAMPLE_SRCS}
)
target_link_libraries(collisions_mw_core PUBLIC
    madrona_mw_core
)

add_executable(collisions_sw
    main.cpp
)
target_link_libraries(collisions_sw PRIVATE
    collisions_sw_core
)

add_executable(collisions_mw
    mw_cpu.cpp
)
target_link_libraries(collisions_mw PRIVATE
    collisions_mw_core
)
