set(EXAMPLE_SRCS
    simple.hpp simple.cpp
)

add_library(simple_taskgraph_core STATIC
    ${EXAMPLE_SRCS}
)
target_link_libraries(simple_taskgraph_core PUBLIC
    madrona_mw_core
)

add_library(simple_taskgraph_init STATIC
    init.cpp init.hpp
)

target_link_libraries(simple_taskgraph_init PRIVATE 
    madrona_common
)

add_executable(simple_taskgraph_cpu
    mw_cpu.cpp
)
target_link_libraries(simple_taskgraph_cpu PRIVATE
    simple_taskgraph_core simple_taskgraph_init
)

if (TARGET madrona_mw_gpu)
    madrona_build_compile_defns(
        OUT_TARGET
            simple_taskgraph_gpu_srcs
        SOURCES_DEFN
            SIMPLE_TASKGRAPH_SRC_LIST
        FLAGS_DEFN
            SIMPLE_TASKGRAPH_COMPILE_FLAGS
        SRCS 
            ${EXAMPLE_SRCS} gpu_entry.cpp
    )

    add_executable(simple_taskgraph_gpu
        gpu.cpp
    )

    target_link_libraries(simple_taskgraph_gpu PRIVATE
        simple_taskgraph_core simple_taskgraph_init madrona_mw_gpu simple_taskgraph_gpu_srcs)
endif()
