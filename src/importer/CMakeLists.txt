add_library(madrona_json STATIC
    ${MADRONA_INC_DIR}/json.hpp json.cpp
)

target_link_libraries(madrona_json
    PUBLIC
        simdjson::simdjson
    PRIVATE
        madrona_common
)

add_library(madrona_importer STATIC
    ${MADRONA_INC_DIR}/importer.hpp importer.cpp
    #gltf.hpp gltf.cpp
    obj.hpp obj.cpp
)

target_link_libraries(madrona_importer PRIVATE
    madrona_common
    madrona_json
    meshoptimizer
    fast_float
)
