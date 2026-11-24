set(OPENBUENOS_DIR "" CACHE PATH "ggml-edgetpu: path to libopenbuenos root or install prefix")
option(GGML_EDGETPU_ALLOW_MOCK_SHIM "Allow built-in mock shim when libopenbuenos is not found" OFF)

if (NOT OPENBUENOS_DIR)
    if (DEFINED ENV{OPENBUENOS_DIR} AND NOT "$ENV{OPENBUENOS_DIR}" STREQUAL "")
        set(OPENBUENOS_DIR "$ENV{OPENBUENOS_DIR}")
    else()
        get_filename_component(_ob_default "${CMAKE_CURRENT_LIST_DIR}/../../../../../libopenbuenos" ABSOLUTE)
        get_filename_component(_ob_sibling "${CMAKE_CURRENT_LIST_DIR}/../libopenbuenos" ABSOLUTE)
        if (EXISTS "${_ob_default}/include/openbuenos.h")
            set(OPENBUENOS_DIR "${_ob_default}")
        elseif (EXISTS "${_ob_sibling}/include/openbuenos.h")
            set(OPENBUENOS_DIR "${_ob_sibling}")
        endif()
    endif()
endif()

set(_ob_target "")
set(_ob_lib "")
set(_ob_inc "")

if (OPENBUENOS_DIR)
    if (ANDROID)
        set(_ob_candidates
            "${OPENBUENOS_DIR}/build/android-arm64/libopenbuenos.a"
            "${OPENBUENOS_DIR}/lib/android-arm64/libopenbuenos.a"
            "${OPENBUENOS_DIR}/build/libopenbuenos.a")
    else()
        set(_ob_candidates
            "${OPENBUENOS_DIR}/build/linux-arm64/libopenbuenos.a"
            "${OPENBUENOS_DIR}/build/host/libopenbuenos.a"
            "${OPENBUENOS_DIR}/lib/linux-arm64/libopenbuenos.a"
            "${OPENBUENOS_DIR}/lib/host/libopenbuenos.a"
            "${OPENBUENOS_DIR}/lib/aarch64-linux-gnu/libopenbuenos.a"
            "${OPENBUENOS_DIR}/lib/libopenbuenos.a"
            "${OPENBUENOS_DIR}/lib64/libopenbuenos.a"
            "${OPENBUENOS_DIR}/build/libopenbuenos.a")
    endif()
    foreach(_cand IN LISTS _ob_candidates)
        if (EXISTS "${_cand}")
            set(_ob_lib "${_cand}")
            break()
        endif()
    endforeach()
    if (EXISTS "${OPENBUENOS_DIR}/include/openbuenos.h")
        set(_ob_inc "${OPENBUENOS_DIR}/include")
    endif()
endif()

if (NOT _ob_lib AND NOT ANDROID)
    find_package(openbuenos CONFIG QUIET)
    if (TARGET openbuenos::openbuenos)
        set(_ob_target openbuenos::openbuenos)
    else()
        find_library(_ob_sys_lib NAMES openbuenos
            PATHS /usr /usr/local
            PATH_SUFFIXES lib lib64 lib/aarch64-linux-gnu)
        find_path(_ob_sys_inc NAMES openbuenos.h
            PATHS /usr /usr/local
            PATH_SUFFIXES include)
        if (_ob_sys_lib AND _ob_sys_inc)
            set(_ob_lib "${_ob_sys_lib}")
            set(_ob_inc "${_ob_sys_inc}")
        endif()
    endif()
endif()

set(GGML_EDGETPU_SOURCES ggml-edgetpu.cpp)

if (_ob_target)
    message(STATUS "ggml-edgetpu: linking installed CMake target ${_ob_target}")
elseif (_ob_lib)
    message(STATUS "ggml-edgetpu: linking libopenbuenos library: ${_ob_lib}")
elseif (GGML_EDGETPU_ALLOW_MOCK_SHIM)
    message(STATUS "ggml-edgetpu: libopenbuenos not found; compiling built-in mock shim (GGML_EDGETPU_ALLOW_MOCK_SHIM=ON)")
    list(APPEND GGML_EDGETPU_SOURCES openbuenos_shim.c)
else()
    message(FATAL_ERROR "ggml-edgetpu: libopenbuenos not found (install omamobile-freetpu, set -DOPENBUENOS_DIR=<path>, or pass -DGGML_EDGETPU_ALLOW_MOCK_SHIM=ON)")
endif()

ggml_add_backend_library(ggml-edgetpu ${GGML_EDGETPU_SOURCES})

if (_ob_target)
    target_link_libraries(ggml-edgetpu PRIVATE ${_ob_target})
    target_compile_definitions(ggml-edgetpu PRIVATE GGML_EDGETPU_LINKED_LIBOPENBUENOS=1)
elseif (_ob_lib)
    if (NOT _ob_inc)
        message(FATAL_ERROR "ggml-edgetpu: openbuenos.h not found alongside ${_ob_lib} (set -DOPENBUENOS_DIR=<path>)")
    endif()
    target_include_directories(ggml-edgetpu PRIVATE "${_ob_inc}")
    target_link_libraries(ggml-edgetpu PRIVATE "${_ob_lib}")
    target_compile_definitions(ggml-edgetpu PRIVATE GGML_EDGETPU_LINKED_LIBOPENBUENOS=1)
else()
    if (_ob_inc)
        target_include_directories(ggml-edgetpu PRIVATE "${_ob_inc}")
    elseif (OPENBUENOS_DIR AND EXISTS "${OPENBUENOS_DIR}/include/openbuenos.h")
        target_include_directories(ggml-edgetpu PRIVATE "${OPENBUENOS_DIR}/include")
    else()
        message(FATAL_ERROR "ggml-edgetpu: openbuenos.h not found for mock shim (set -DOPENBUENOS_DIR=<path>)")
    endif()
    target_compile_definitions(ggml-edgetpu PRIVATE GGML_EDGETPU_LINKED_LIBOPENBUENOS=0)
endif()

if (NOT WIN32)
    target_link_libraries(ggml-edgetpu PRIVATE m)
    if (NOT ANDROID)
        find_package(Threads REQUIRED)
        target_link_libraries(ggml-edgetpu PRIVATE Threads::Threads)
    endif()
endif()
