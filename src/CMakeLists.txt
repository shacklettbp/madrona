add_subdirectory(common)
add_subdirectory(core)
add_subdirectory(physics)

add_subdirectory(render)
add_subdirectory(mw_cpu)

if (CUDAToolkit_FOUND)
    add_subdirectory(mw_gpu)
endif ()

if (Python_FOUND)
    add_subdirectory(python)
endif ()
