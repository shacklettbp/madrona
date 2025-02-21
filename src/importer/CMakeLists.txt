set(MADRONA_GLTF_SUPPORT ON CACHE BOOL "")

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
    obj.hpp obj.cpp
    stl.hpp stl.cpp
    urdf.cpp
    stb_read.cpp img.cpp
)

if (MADRONA_GLTF_SUPPORT)
    list(APPEND IMPORTER_SOURCES
         gltf.hpp gltf.cpp
    )
endif()

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
    stb
    tinyxml2
)

if (MADRONA_GLTF_SUPPORT)
    target_compile_definitions(madrona_importer PRIVATE
        MADRONA_GLTF_SUPPORT=1
    )
endif()

if (MADRONA_USD_SUPPORT)
    target_link_libraries(madrona_importer PRIVATE
        madrona_tinyusdz
    )

    target_compile_definitions(madrona_importer PRIVATE
        MADRONA_USD_SUPPORT=1
    )

endif()
