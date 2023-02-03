set(MADRONA_EXTERNAL_DIR "" CACHE
    STRING "External Directory to use for madrona sources")

if (MADRONA_EXTERNAL_DIR)
    set(MADRONA_DIR "${MADRONA_EXTERNAL_DIR}")
else()
    function(madrona_default_dir)
        get_filename_component(MADRONA_DIR
            "${CMAKE_CURRENT_FUNCTION_LIST_DIR}" DIRECTORY)
        set(MADRONA_DIR "${MADRONA_DIR}" PARENT_SCOPE)
    endfunction()

    madrona_default_dir()
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${MADRONA_DIR}/cmake")

include(build_type)
