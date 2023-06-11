add_library(madrona_render_shader SHARED
    ${MADRONA_INC_DIR}/render/shader_compiler.hpp compiler.cpp
)

target_link_libraries(madrona_render_shader PRIVATE
    madrona_common
    madrona_dxcompiler
    spv_reflect
)
