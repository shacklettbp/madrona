set(FVS_CORE_SRCS
    fvs.hpp fvs.cpp
)

add_library(fvs_sw_core STATIC
    ${FVS_CORE_SRCS}
)
target_link_libraries(fvs_sw_core PUBLIC
    madrona_core
)

add_library(fvs_mw_core STATIC
    ${FVS_CORE_SRCS}
)
target_link_libraries(fvs_mw_core PUBLIC
    madrona_mw_core
)

add_executable(fvs_sw
    main.cpp
)
target_link_libraries(fvs_sw PRIVATE
    fvs_sw_core
)

add_executable(fvs_mw
    mw_cpu.cpp
)
target_link_libraries(fvs_mw PRIVATE
    fvs_mw_core
)
