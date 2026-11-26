cmake_minimum_required(VERSION 3.16)
project(freetile LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(freetile STATIC
    src/tile_ir.cpp
    src/mosaic_emitter.cpp
    src/buenos_emitter.cpp
    src/buenos_splines.cpp
    src/flash_attn_planner.cpp
    src/malibu_profile.cpp
)
target_include_directories(freetile PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_options(freetile PRIVATE -Wall -Wextra -Wpedantic -Werror)

enable_testing()
add_executable(test_freetile
    tests/test_freetile.cpp
)
target_link_libraries(test_freetile PRIVATE freetile)
target_compile_options(test_freetile PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_test(NAME freetile_unit_tests COMMAND test_freetile)

add_executable(test_splines_and_flash_attn
    tests/test_splines_and_flash_attn.cpp
)
target_link_libraries(test_splines_and_flash_attn PRIVATE freetile)
target_compile_options(test_splines_and_flash_attn PRIVATE
    -Wall -Wextra -Wpedantic -Werror -fno-exceptions -fno-rtti)
add_test(NAME freetile_splines_flash_attn_tests COMMAND test_splines_and_flash_attn)

option(FREETILE_BUILD_BUENOS_PARITY_TEST "Build bit-exact parity test against sibling libopenbuenos" ON)
option(FREETILE_BUILD_DIGITAL_TWIN "Build freetile-twin digital twin library, CLI, and tests" ON)

if((FREETILE_BUILD_BUENOS_PARITY_TEST OR FREETILE_BUILD_DIGITAL_TWIN)
   AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../libopenbuenos/CMakeLists.txt")
    set(OPENBUENOS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    add_subdirectory(
        "${CMAKE_CURRENT_SOURCE_DIR}/../libopenbuenos"
        "${CMAKE_CURRENT_BINARY_DIR}/openbuenos_ref"
        EXCLUDE_FROM_ALL
    )

    if(FREETILE_BUILD_BUENOS_PARITY_TEST)
        add_executable(test_buenos_parity
            tests/test_buenos_parity.cpp
        )
        target_link_libraries(test_buenos_parity PRIVATE freetile openbuenos)
        target_include_directories(test_buenos_parity SYSTEM PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/../libopenbuenos/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/../libopenbuenos/src"
            "${CMAKE_CURRENT_SOURCE_DIR}/../docs/baremetal-vii"
            "${CMAKE_CURRENT_SOURCE_DIR}/../docs/kernel-abi"
        )
        target_compile_options(test_buenos_parity PRIVATE -Wall -Wextra -Wpedantic -Werror)
        add_test(NAME freetile_buenos_parity_tests COMMAND test_buenos_parity)
    endif()

    if(FREETILE_BUILD_DIGITAL_TWIN)
        add_library(freetile_twin STATIC
            src/twin.cpp
        )
        target_link_libraries(freetile_twin PUBLIC freetile PRIVATE openbuenos)
        target_include_directories(freetile_twin PUBLIC
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
        )
        target_include_directories(freetile_twin SYSTEM PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/../libopenbuenos/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/../libopenbuenos/src"
            "${CMAKE_CURRENT_SOURCE_DIR}/../docs/baremetal-vii"
            "${CMAKE_CURRENT_SOURCE_DIR}/../docs/kernel-abi"
        )
        target_compile_options(freetile_twin PRIVATE
            -Wall -Wextra -Wpedantic -Werror -fno-exceptions -fno-rtti)

        add_executable(freetile-twin
            tools/freetile_twin_cli.cpp
        )
        target_link_libraries(freetile-twin PRIVATE freetile_twin freetile openbuenos)
        target_compile_options(freetile-twin PRIVATE
            -Wall -Wextra -Wpedantic -Werror -fno-exceptions -fno-rtti)

        add_executable(test_freetile_twin
            tests/test_freetile_twin.cpp
        )
        target_link_libraries(test_freetile_twin PRIVATE freetile_twin freetile openbuenos)
        target_compile_options(test_freetile_twin PRIVATE
            -Wall -Wextra -Wpedantic -Werror -fno-exceptions -fno-rtti)
        add_test(NAME freetile_twin_tests COMMAND test_freetile_twin)
    endif()
endif()
