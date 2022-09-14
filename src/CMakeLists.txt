add_subdirectory(common)
add_subdirectory(core)

if (CUDAToolkit_FOUND)
    add_subdirectory(mw_gpu)
endif ()
