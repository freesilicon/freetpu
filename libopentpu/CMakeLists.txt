cmake_minimum_required(VERSION 3.18)
project(opentpu VERSION 0.1.0 LANGUAGES C CXX)

option(OPENTPU_BUILD_TESTS "Build libopentpu host unit tests" ON)
option(OPENTPU_BUILD_TOOLS "Build libopentpu VFIO diagnostic tools" ON)

add_library(opentpu SHARED
  src/ot_vfio.cpp
  src/ot_device.cpp
  src/ot_pjrt.cpp
)

add_library(opentpu::opentpu ALIAS opentpu)

target_compile_features(opentpu PRIVATE cxx_std_17)
target_compile_options(opentpu PRIVATE
  -Wall -Wextra -Werror -fno-exceptions -fno-rtti
)

target_include_directories(opentpu
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/openxla
)

if (UNIX AND NOT APPLE)
  target_link_libraries(opentpu PUBLIC m pthread dl)
else()
  target_link_libraries(opentpu PUBLIC m)
endif()

if (OPENTPU_BUILD_TESTS)
  add_executable(opentpu_host_test tests/opentpu_host_test.cpp)
  target_link_libraries(opentpu_host_test PRIVATE opentpu)
  target_include_directories(opentpu_host_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/openxla
  )
  target_compile_features(opentpu_host_test PRIVATE cxx_std_17)
  target_compile_options(opentpu_host_test PRIVATE -Wall -Wextra -Werror -fno-exceptions -fno-rtti)
  enable_testing()
  add_test(NAME opentpu_host_test COMMAND opentpu_host_test)
endif()

if (OPENTPU_BUILD_TOOLS AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_executable(vfio_m1_probe tools/vfio_m1_probe.cpp)
  target_compile_features(vfio_m1_probe PRIVATE cxx_std_17)
  target_compile_options(vfio_m1_probe PRIVATE -Wall -Wextra)
endif()
