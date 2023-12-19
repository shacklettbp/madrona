set(MADRONA_CORE_SRCS
    ${MADRONA_INC_DIR}/fwd.hpp
    ${MADRONA_INC_DIR}/exec_mode.hpp
    #${MADRONA_INC_DIR}/job.hpp ${MADRONA_INC_DIR}/job.inl job.cpp
    ${MADRONA_INC_DIR}/state.hpp ${MADRONA_INC_DIR}/state.inl state.cpp
    ${MADRONA_INC_DIR}/context.hpp ${MADRONA_INC_DIR}/context.inl context.cpp
    ${MADRONA_INC_DIR}/components.hpp base.cpp
    #${MADRONA_INC_DIR}/hash.hpp
    #${MADRONA_INC_DIR}/platform_utils.hpp
    #${MADRONA_INC_DIR}/platform_utils.inl
    #    platform_utils.cpp
)

add_library(madrona_core STATIC
    ${MADRONA_CORE_SRCS}
)

target_link_libraries(madrona_core
    PUBLIC 
        madrona_common
)

target_compile_definitions(madrona_core
    PUBLIC
        -DMADRONA_USE_TASK_GRAPH=1
)

add_library(madrona_mw_core STATIC
    ${MADRONA_CORE_SRCS}
    ${MADRONA_INC_DIR}/taskgraph.hpp ${MADRONA_INC_DIR}/taskgraph.inl taskgraph.cpp
)

target_compile_definitions(madrona_mw_core
    PUBLIC
        -DMADRONA_MW_MODE=1
        -DMADRONA_USE_TASK_GRAPH=1
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
