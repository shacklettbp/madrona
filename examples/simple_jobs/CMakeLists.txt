set(EXAMPLE_SRCS
    simple.hpp simple.cpp
)

add_library(simple_ex_sw_core STATIC
    ${EXAMPLE_SRCS}
)
target_link_libraries(simple_ex_sw_core PUBLIC
    madrona_core
)

add_library(simple_ex_mw_core STATIC
    ${EXAMPLE_SRCS}
)
target_link_libraries(simple_ex_mw_core PUBLIC
    madrona_mw_core
)

add_library(simple_ex_init STATIC
    init.cpp init.hpp
)

target_link_libraries(simple_ex_init PRIVATE 
    madrona_mw_gpu
)

add_executable(simple_ex_sw
    main.cpp
)
target_link_libraries(simple_ex_sw PRIVATE
    simple_ex_sw_core simple_ex_init
)

add_executable(simple_ex_mw
    mw_cpu.cpp
)
target_link_libraries(simple_ex_mw PRIVATE
    simple_ex_mw_core simple_ex_init
)

if (TARGET madrona_mw_gpu)
    madrona_build_compile_defns(
        OUT_TARGET
            simple_ex_gpu_srcs
        SOURCES_DEFN
            SIMPLE_EX_SRC_LIST
        FLAGS_DEFN
            SIMPLE_EX_COMPILE_FLAGS
        SRCS 
            ${EXAMPLE_SRCS} gpu_entry.cpp
    )

    add_executable(simple_ex_gpu
        gpu.cpp
    )

    target_link_libraries(simple_ex_gpu PRIVATE
        simple_ex_mw_core simple_ex_init madrona_mw_gpu simple_ex_gpu_srcs)
endif()
