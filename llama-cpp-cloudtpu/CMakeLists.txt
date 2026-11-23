cmake_minimum_required(VERSION 3.16)

if (COMMAND ggml_add_backend_library)
    include(CheckCXXCompilerFlag)
    set(CLOUDTPU_PJRT_INCLUDE_DIR "" CACHE PATH "ggml-cloudtpu: path to directory containing pjrt_c_api.h")
    if (NOT CLOUDTPU_PJRT_INCLUDE_DIR)
        if (EXISTS "${CMAKE_CURRENT_LIST_DIR}/third_party/openxla/pjrt_c_api.h")
            set(CLOUDTPU_PJRT_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/third_party/openxla")
        elseif (EXISTS "${CMAKE_CURRENT_LIST_DIR}/../libopentpu/third_party/openxla/pjrt_c_api.h")
            set(CLOUDTPU_PJRT_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../libopentpu/third_party/openxla")
        elseif (DEFINED ENV{TPU_PJRT_IR_DIR} AND EXISTS "$ENV{TPU_PJRT_IR_DIR}/pjrt_c_api.h")
            set(CLOUDTPU_PJRT_INCLUDE_DIR "$ENV{TPU_PJRT_IR_DIR}")
        endif()
    endif()

    if (NOT EXISTS "${CLOUDTPU_PJRT_INCLUDE_DIR}/pjrt_c_api.h")
        message(FATAL_ERROR "ggml-cloudtpu: pjrt_c_api.h not found (set -DCLOUDTPU_PJRT_INCLUDE_DIR=<path>)")
    endif()

    ggml_add_backend_library(ggml-cloudtpu
        ggml-cloudtpu.cpp
        pjrt_runtime.cpp
        stablehlo_emitter.cpp
    )

    target_compile_features(ggml-cloudtpu PRIVATE cxx_std_17)
    target_include_directories(ggml-cloudtpu PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
    target_include_directories(ggml-cloudtpu SYSTEM PRIVATE "${CLOUDTPU_PJRT_INCLUDE_DIR}")

    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        check_cxx_compiler_flag("-Wno-changes-meaning" CLOUDTPU_HAS_NO_CHANGES_MEANING)
        if (CLOUDTPU_HAS_NO_CHANGES_MEANING)
            target_compile_options(ggml-cloudtpu PRIVATE -Wno-changes-meaning)
        endif()
    endif()

    if (NOT WIN32)
        target_link_libraries(ggml-cloudtpu PRIVATE ${CMAKE_DL_LIBS} m)
        find_package(Threads REQUIRED)
        target_link_libraries(ggml-cloudtpu PRIVATE Threads::Threads)
    endif()
else()
    project(llama_cpp_cloudtpu_host LANGUAGES CXX)

    include(CheckCXXCompilerFlag)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)

    find_package(Threads REQUIRED)

    set(CLOUDTPU_PJRT_INCLUDE_DIR "" CACHE PATH "ggml-cloudtpu: path to directory containing pjrt_c_api.h")
    if (NOT CLOUDTPU_PJRT_INCLUDE_DIR)
        if (EXISTS "${CMAKE_CURRENT_LIST_DIR}/../libopentpu/third_party/openxla/pjrt_c_api.h")
            set(CLOUDTPU_PJRT_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/../libopentpu/third_party/openxla")
        elseif (EXISTS "${CMAKE_CURRENT_LIST_DIR}/third_party/openxla/pjrt_c_api.h")
            set(CLOUDTPU_PJRT_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/third_party/openxla")
        endif()
    endif()
    if (NOT EXISTS "${CLOUDTPU_PJRT_INCLUDE_DIR}/pjrt_c_api.h")
        message(FATAL_ERROR "ggml-cloudtpu: pjrt_c_api.h not found (set -DCLOUDTPU_PJRT_INCLUDE_DIR=<path>)")
    endif()

    add_library(cloudtpu_core STATIC
        pjrt_runtime.cpp
        stablehlo_emitter.cpp
    )
    target_include_directories(cloudtpu_core PUBLIC "${CMAKE_CURRENT_LIST_DIR}")
    target_include_directories(cloudtpu_core SYSTEM PUBLIC "${CLOUDTPU_PJRT_INCLUDE_DIR}")
    if (NOT WIN32)
        target_link_libraries(cloudtpu_core PUBLIC ${CMAKE_DL_LIBS} Threads::Threads)
    endif()
    if (NOT MSVC)
        target_compile_options(cloudtpu_core PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        check_cxx_compiler_flag("-Wno-changes-meaning" CLOUDTPU_HAS_NO_CHANGES_MEANING)
        if (CLOUDTPU_HAS_NO_CHANGES_MEANING)
            target_compile_options(cloudtpu_core PUBLIC -Wno-changes-meaning)
        endif()
    endif()

    enable_testing()
    add_executable(test_cloudtpu_host
        tests/test_cloudtpu_host.cpp
    )
    target_link_libraries(test_cloudtpu_host PRIVATE cloudtpu_core)
    if (NOT MSVC)
        target_compile_options(test_cloudtpu_host PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME cloudtpu_host_tests COMMAND test_cloudtpu_host)
endif()
