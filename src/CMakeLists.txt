set(SOURCES
    functionality.cpp
)

#add_library(madrona_core
#    ${SOURCES}
#)

if (CUDAToolkit_FOUND)
    add_subdirectory(train)
endif ()
