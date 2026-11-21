set(OPENBUENOS_DIR "" CACHE PATH "ggml-edgetpu: path to libopenbuenos root directory")

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

set(_ob_lib "")
if (OPENBUENOS_DIR)
    if (ANDROID)
        set(_ob_candidates
            "${OPENBUENOS_DIR}/build/android-arm64/libopenbuenos.a"
            "${OPENBUENOS_DIR}/lib/android-arm64/libopenbuenos.a"
            "${OPENBUENOS_DIR}/build/libopenbuenos.a")
    else()
        set(_ob_candidates
            "${OPENBUENOS_DIR}/build/host/libopenbuenos.a"
            "${OPENBUENOS_DIR}/lib/host/libopenbuenos.a"
            "${OPENBUENOS_DIR}/build/libopenbuenos.a")
    endif()
    foreach(_cand IN LISTS _ob_candidates)
        if (EXISTS "${_cand}")
            set(_ob_lib "${_cand}")
            break()
        endif()
    endforeach()
endif()

set(GGML_EDGETPU_SOURCES ggml-edgetpu.cpp)

if (_ob_lib)
    message(STATUS "ggml-edgetpu: linking libopenbuenos archive: ${_ob_lib}")
else()
    message(STATUS "ggml-edgetpu: libopenbuenos.a not found under '${OPENBUENOS_DIR}'; compiling built-in mock shim")
    list(APPEND GGML_EDGETPU_SOURCES openbuenos_shim.c)
endif()

ggml_add_backend_library(ggml-edgetpu ${GGML_EDGETPU_SOURCES})

if (OPENBUENOS_DIR AND EXISTS "${OPENBUENOS_DIR}/include/openbuenos.h")
    target_include_directories(ggml-edgetpu PRIVATE "${OPENBUENOS_DIR}/include")
    message(STATUS "ggml-edgetpu: using openbuenos.h from ${OPENBUENOS_DIR}/include")
else()
    message(FATAL_ERROR "ggml-edgetpu: openbuenos.h not found (set -DOPENBUENOS_DIR=<path>)")
endif()

if (_ob_lib)
    target_link_libraries(ggml-edgetpu PRIVATE "${_ob_lib}")
    target_compile_definitions(ggml-edgetpu PRIVATE GGML_EDGETPU_LINKED_LIBOPENBUENOS=1)
else()
    target_compile_definitions(ggml-edgetpu PRIVATE GGML_EDGETPU_LINKED_LIBOPENBUENOS=0)
endif()

if (NOT WIN32)
    target_link_libraries(ggml-edgetpu PRIVATE m)
    if (NOT ANDROID)
        target_link_libraries(ggml-edgetpu PRIVATE Threads::Threads)
    endif()
endif()
