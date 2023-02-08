add_library(madrona_python_utils SHARED
     utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/madrona/python.hpp
)

target_link_libraries(madrona_python_utils
    PUBLIC
        madrona_hdrs
    PRIVATE
        madrona_common
)

if (TARGET madrona_cuda)
    target_link_libraries(madrona_python_utils
        PUBLIC
            CUDA::cudart
        PRIVATE
            madrona_cuda
    )
endif ()

function(madrona_python_module module_name)
    nanobind_add_module(${module_name} NB_SHARED
        ${ARGN}
    )

    target_link_libraries(${module_name} PRIVATE
        madrona_python_utils
    )
     
    set_target_properties(madrona_python PROPERTIES EXCLUDE_FROM_ALL FALSE)
endfunction()

madrona_python_module(madrona_python
    bindings.cpp
)

if (FRONTEND_GCC OR FRONTEND_CLANG)
    set_source_files_properties(bindings.cpp PROPERTIES COMPILE_OPTIONS
        "-frtti;-fexceptions")
elseif (FRONTEND_MSVC)
    set_source_files_properties(bindings.cpp PROPERTIES COMPILE_OPTIONS
        "/GR+")
endif ()

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
