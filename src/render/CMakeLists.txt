add_subdirectory(shader)

if (MADRONA_LINUX OR MADRONA_WINDOWS OR MADRONA_ANDROID OR MADRONA_MACOS)
    add_subdirectory(vk)
endif()

if (MADRONA_MACOS OR MADRONA_IOS)
    add_subdirectory(metal)
endif ()

#add_subdirectory(core)
