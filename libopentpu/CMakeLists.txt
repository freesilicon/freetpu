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
  target_link_options(opentpu PRIVATE -static-libstdc++ -static-libgcc -Wl,--as-needed -Wl,--exclude-libs,ALL)
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

if (OPENTPU_BUILD_TOOLS)
  add_executable(opentpu_vfio_smoke tools/opentpu_vfio_smoke.cpp)
  target_link_libraries(opentpu_vfio_smoke PRIVATE opentpu)
  target_compile_features(opentpu_vfio_smoke PRIVATE cxx_std_17)
  target_compile_options(opentpu_vfio_smoke PRIVATE -Wall -Wextra -Werror -fno-exceptions -fno-rtti)

  add_executable(opentpu_matmul_proof tools/opentpu_matmul_proof.cpp)
  add_dependencies(opentpu_matmul_proof opentpu)
  target_include_directories(opentpu_matmul_proof PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/openxla
  )
  target_compile_features(opentpu_matmul_proof PRIVATE cxx_std_17)
  target_compile_options(opentpu_matmul_proof PRIVATE -Wall -Wextra -Werror -fno-exceptions -fno-rtti)
  if (UNIX AND NOT APPLE)
    target_link_libraries(opentpu_matmul_proof PRIVATE dl m)
    target_link_options(opentpu_matmul_proof PRIVATE -static-libstdc++ -static-libgcc -Wl,--as-needed)
  else()
    target_link_libraries(opentpu_matmul_proof PRIVATE m)
  endif()

  if (OPENTPU_BUILD_TESTS)
    add_test(NAME opentpu_vfio_smoke_mock COMMAND opentpu_vfio_smoke --mock)
    add_test(NAME opentpu_matmul_proof_mock
             COMMAND opentpu_matmul_proof --mock --lib $<TARGET_FILE:opentpu>)
  endif()

  if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(vfio_m1_probe tools/vfio_m1_probe.cpp)
    target_compile_features(vfio_m1_probe PRIVATE cxx_std_17)
    target_compile_options(vfio_m1_probe PRIVATE -Wall -Wextra)
  endif()
endif()
