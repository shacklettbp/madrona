add_library(madrona_render_core STATIC
    ${MADRONA_INC_DIR}/render/graph.hpp graph.cpp
)

target_link_libraries(madrona_render_core PUBLIC
    madrona_common
)

if (TARGET madrona_render_vk)
    target_link_libraries(madrona_render_core PRIVATE madrona_render_vk)
elseif (TARGET madrona_render_metal)
    target_link_libraries(madrona_render_core PRIVATE madrona_render_metal)
endif ()
