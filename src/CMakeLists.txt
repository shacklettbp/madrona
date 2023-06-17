add_subdirectory(common)
add_subdirectory(core)
add_subdirectory(importer)
add_subdirectory(physics)

add_subdirectory(render)

add_subdirectory(mw)

if (Python_FOUND)
    add_subdirectory(python)
endif ()

add_subdirectory(viz)
