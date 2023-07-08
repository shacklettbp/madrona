add_library(madrona_json STATIC
    ${MADRONA_INC_DIR}/json.hpp json.cpp
)

target_link_libraries(madrona_json
    PUBLIC
        simdjson::simdjson
    PRIVATE
        madrona_common
)

set(IMPORTER_SOURCES
    ${MADRONA_INC_DIR}/importer.hpp importer.cpp
    gltf.hpp gltf.cpp
    obj.hpp obj.cpp
)

if (MADRONA_USD_SUPPORT)
    list(APPEND IMPORTER_SOURCES
        usd.hpp usd.cpp
    )
endif()

add_library(madrona_importer STATIC
    ${IMPORTER_SOURCES}
)

target_link_libraries(madrona_importer PRIVATE
    madrona_common
    madrona_json
    meshoptimizer
    fast_float
)

if (MADRONA_USD_SUPPORT)
    target_link_libraries(madrona_importer PRIVATE
        madrona_tinyusdz
    )

    target_compile_definitions(madrona_importer PRIVATE
        MADRONA_USD_SUPPORT=1
    )

endif()
