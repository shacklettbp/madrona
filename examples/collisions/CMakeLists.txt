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

add_library(collisions_gpu_core STATIC
    ${COLLISION_EXAMPLE_SRCS}
)
target_link_libraries(collisions_gpu_core PUBLIC
    madrona_mw_gpu
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

if (TARGET madrona_mw_gpu)
    madrona_build_compile_defns(
        OUT_TARGET
            collisions_gpu_srcs
        SOURCES_DEFN
            COLLISIONS_SRC_LIST
        FLAGS_DEFN
            COLLISIONS_COMPILE_FLAGS
        SRC_TARGETS
            collisions_gpu_core
    )

    add_executable(collisions_gpu
        gpu.cpp
    )

    target_link_libraries(collisions_gpu PRIVATE
        collisions_mw_core madrona_mw_gpu collisions_gpu_srcs)
endif()
