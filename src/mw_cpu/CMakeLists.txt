add_library(madrona_mw_cpu STATIC
    ${MADRONA_INC_DIR}/mw_cpu.hpp ${MADRONA_INC_DIR}/mw_cpu.inl cpu_exec.cpp
)

target_link_libraries(madrona_mw_cpu
    PUBLIC
        madrona_mw_core
    PRIVATE
        madrona_batch_render
)
