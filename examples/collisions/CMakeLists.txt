set(FVS_CORE_SRCS
    CollisionExample.hpp CollisionExample.inl CollisionExample.cpp
)

add_library(CollisionExample_sw_core STATIC
    ${FVS_CORE_SRCS}
)
target_link_libraries(CollisionExample_sw_core PUBLIC
    madrona_core
)

add_library(CollisionExample_mw_core STATIC
    ${FVS_CORE_SRCS}
)
target_link_libraries(CollisionExample_mw_core PUBLIC
    madrona_mw_core
)

add_executable(CollisionExample
    main.cpp
)
target_link_libraries(CollisionExample PRIVATE
    CollisionExample_sw_core
)

add_executable(CollisionExample_mw
    mw_cpu.cpp
)
target_link_libraries(CollisionExample_mw PRIVATE
    CollisionExample_mw_core
)
