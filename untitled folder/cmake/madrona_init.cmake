set(MADRONA_EXTERNAL_DIR "" CACHE
    STRING "External Directory to use for madrona sources")

if (MADRONA_EXTERNAL_DIR)
    if (CMAKE_PROJECT_NAME STREQUAL "madrona" AND PROJECT_IS_TOP_LEVEL)
        message(FATAL_ERROR "Cannot use MADRONA_EXTERNAL_DIR for standalone builds")
    endif()
    set(MADRONA_DIR "${MADRONA_EXTERNAL_DIR}")
else()
    function(madrona_default_dir)
        get_filename_component(MADRONA_DIR
            "${CMAKE_CURRENT_FUNCTION_LIST_DIR}" DIRECTORY)
        set(MADRONA_DIR "${MADRONA_DIR}" PARENT_SCOPE)
    endfunction()

    madrona_default_dir()
    unset(madrona_default_dir)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${MADRONA_DIR}/cmake")

include(build_type)

if (NOT CMAKE_PROJECT_NAME AND NOT CMAKE_TOOLCHAIN_FILE AND NOT WIN32)
    include("${MADRONA_DIR}/external/madrona-toolchain/cmake/set_toolchain.cmake")
endif()
