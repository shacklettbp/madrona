add_library(madrona_render_shader_compiler SHARED
    ${MADRONA_INC_DIR}/render/shader_compiler.hpp compiler.cpp
)

target_link_libraries(madrona_render_shader_compiler PRIVATE
    madrona_common
    madrona_dxcompiler
    spv_reflect
)
