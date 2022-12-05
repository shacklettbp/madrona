nanobind_add_module(madrona_python NB_STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/madrona/python.hpp
    python.cpp
)

# HACK: remove the warnings from the nanobind library. Because of
# how the nanobind cmake setup works, the nanobind library target isn't
# declared until the first call to nanobind_add_module, so all project
# warnings get applied

get_target_property(nanobind_lib_target madrona_python LINK_LIBRARIES)
foreach(lib IN LISTS nanobind_lib_target)
    string(FIND ${lib} "nanobind" pos)
    if (NOT ${pos} EQUAL -1)
        target_compile_options(${lib} PRIVATE "-Wno-everything")
    endif ()
endforeach()

target_link_libraries(madrona_python
    PRIVATE
        madrona_hdrs
        CUDA::cudart
)
