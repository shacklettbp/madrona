add_subdirectory(common)
add_subdirectory(core)
add_subdirectory(physics)
#add_subdirectory(render)

if (CUDAToolkit_FOUND)
    add_subdirectory(mw_gpu)
endif ()
