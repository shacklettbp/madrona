add_library(madrona_render_core STATIC
    ${MADRONA_INC_DIR}/render/graph.hpp ${MADRONA_INC_DIR}/render/graph.inl graph.cpp
    ${MADRONA_INC_DIR}/render/reflection.hpp ${MADRONA_INC_DIR}/render/backend.hpp
    ${MADRONA_INC_DIR}/render/gpu.hpp ${MADRONA_INC_DIR}/render/gpu.inl
    ${MADRONA_INC_DIR}/render/shader.hpp ${MADRONA_INC_DIR}/render/shader.inl
)

target_link_libraries(madrona_render_core PUBLIC
    madrona_common
)

if (TARGET madrona_render_vk)
    target_link_libraries(madrona_render_core PRIVATE madrona_render_vk)
elseif (TARGET madrona_render_metal)
    target_link_libraries(madrona_render_core PRIVATE madrona_render_metal)
endif ()
