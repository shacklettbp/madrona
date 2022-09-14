set(INC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../include/madrona")

set(MADRONA_CORE_SRCS
    ${INC_DIR}/fwd.hpp
    ${INC_DIR}/job.hpp ${INC_DIR}/job.inl job.cpp
    ${INC_DIR}/state.hpp ${INC_DIR}/state.inl state.cpp
    ${INC_DIR}/context.hpp ${INC_DIR}/context.inl context.cpp
    #${INC_DIR}/hash.hpp
    #${INC_DIR}/platform_utils.hpp ${INC_DIR}/platform_utils.inl
    #    platform_utils.cpp
)

add_library(madrona_core STATIC
    ${MADRONA_CORE_SRCS}
)

target_link_libraries(madrona_core
    PUBLIC 
        madrona_common
)

add_library(madrona_mw_core STATIC
    ${MADRONA_CORE_SRCS}
)

target_compile_definitions(madrona_mw_core
    PUBLIC
        -DMADRONA_MW_MODE=1
)

target_link_libraries(madrona_mw_core
    PUBLIC
        madrona_common
)

#add_library(madrona_wm STATIC
#    ${INC_DIR}/window.hpp ${INC_DIR}/window.inl window.cpp
#    ${INC_DIR}/wm.hpp ${INC_DIR}/wm.inl wm.cpp
#    ${INC_DIR}/input_button.hpp 
#)
#
#target_link_libraries(madrona_wm PUBLIC 
#    madrona_core
#    madrona_sdl2
#)
