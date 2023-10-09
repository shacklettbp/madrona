add_library(madrona_python_utils STATIC
     utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/madrona/py/utils.hpp
)

target_link_libraries(madrona_python_utils
    PUBLIC
        madrona_hdrs
)

if (NOT WIN32) #FIX
    target_link_libraries(madrona_python_utils
        PRIVATE
            madrona_libcxx
    )
    target_compile_options(madrona_python_utils
        PRIVATE
            -fno-exceptions
    )
endif()

if (TARGET madrona_cuda)
    target_link_libraries(madrona_python_utils
        PUBLIC
            CUDA::cudart
        PRIVATE
            madrona_cuda
    )
endif ()

get_filename_component(MADRONA_PY_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
set_property(GLOBAL PROPERTY madrona_python_dir ${MADRONA_PY_DIR})

function(madrona_python_add_binding_lib nanobind_lib_name)
    if (TARGET madrona_python_bindings)
        return()
    endif()

    get_property(py_dir GLOBAL PROPERTY madrona_python_dir)

    add_library(madrona_python_bindings OBJECT
        ${py_dir}/bindings.cpp
        ${py_dir}/../../include/madrona/py/bindings.hpp
    )

    # This is also hacky, nanobind doesn't make it easy to declare
    # a static library of helpers built on nanobind for inclusion
    # into other modules.
    target_link_libraries(madrona_python_bindings
        PUBLIC
            ${nanobind_lib_name}
            madrona_hdrs
        PRIVATE
            madrona_err
    )

    if (TARGET madrona_cuda)
        target_link_libraries(madrona_python_bindings PUBLIC
            madrona_cuda
        )
    endif ()
endfunction()

function(madrona_python_get_nanobind_target module_name out_name)
    get_target_property(nanobind_lib_target ${module_name} LINK_LIBRARIES)
    foreach(lib IN LISTS nanobind_lib_target)
        string(FIND ${lib} "nanobind" pos)
        if (${pos} EQUAL -1)
            continue()
        endif()

        set(${out_name} ${lib} PARENT_SCOPE)
        return()
    endforeach()

    message(FATAL_ERROR "Couldn't find nanobind target")
endfunction()

# HACK: remove the warnings from the nanobind library. Because of
# how the nanobind cmake setup works, the nanobind library target isn't
# declared until the first call to nanobind_add_module, so all project
# warnings get applied

function(madrona_python_disable_nanobind_warnings module_name)
    get_property(hack_applied GLOBAL PROPERTY
        madrona_nanobind_hack_applied SET)

    if (NOT hack_applied)
        set_property(GLOBAL PROPERTY madrona_nanobind_hack_applied ON)
    else ()
        return()
    endif()

    if (NOT FRONTEND_MSVC)
        target_compile_options(${nanobind_target_name} PRIVATE
            "-Wno-everything")
    endif()
    
    if (NOT WIN32) #FIX
        target_link_libraries(${nanobind_target_name} PRIVATE
            madrona_libcxx_except
        )
    endif()
endfunction()

function(madrona_python_module module_name)
    nanobind_add_module(${module_name} NB_STATIC
        ${ARGN}
    )

    madrona_python_get_nanobind_target(${module_name} nanobind_target_name)
    madrona_python_disable_nanobind_warnings(${nanobind_target_name})
    madrona_python_add_binding_lib(${nanobind_target_name})

    target_link_libraries(${module_name} PRIVATE
        madrona_python_bindings
        madrona_python_utils
    )

    if (NOT WIN32) #FIX
        target_link_libraries(${module_name} PRIVATE
            madrona_libcxx_except
        )
    endif()
endfunction()
