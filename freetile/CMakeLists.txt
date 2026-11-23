cmake_minimum_required(VERSION 3.16)
project(freetile LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(freetile STATIC
    src/tile_ir.cpp
    src/mosaic_emitter.cpp
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
