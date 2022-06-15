add_subdirectory(core)

if (CUDAToolkit_FOUND)
    add_subdirectory(train)
endif ()
