cmake_minimum_required(VERSION 3.18)
project(openbuenos VERSION 0.1.0 LANGUAGES C CXX)

option(OPENBUENOS_BUILD_TESTS "Build libopenbuenos self-test / host unit test binaries" ON)
option(OPENBUENOS_BUILD_SHARED "Build libopenbuenos shared library for Python ctypes bridge" ON)

set(OPENBUENOS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

set(OPENBUENOS_SOURCES
  src/ob_sha256.cpp
  src/ob_dgc0.cpp
  src/ob_buffer_pool.cpp
  src/ob_worker_pool.cpp
  src/ob_device.cpp
  src/ob_graph.cpp
  src/ob_gemm_emitter.cpp
  src/ob_gemm.cpp
)

add_library(openbuenos STATIC ${OPENBUENOS_SOURCES})

add_library(openbuenos::openbuenos ALIAS openbuenos)

target_compile_features(openbuenos PRIVATE cxx_std_17)
target_compile_options(openbuenos PRIVATE
  -Wall -Wextra -Werror -fno-exceptions -fno-rtti
)

target_include_directories(openbuenos
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OPENBUENOS_ROOT}/docs/baremetal-vii
    ${OPENBUENOS_ROOT}/docs/kernel-abi
)

if (UNIX AND NOT APPLE)
  if (ANDROID)
    target_link_libraries(openbuenos PUBLIC m)
  else()
    target_link_libraries(openbuenos PUBLIC m pthread)
  endif()
elseif (APPLE)
  target_link_libraries(openbuenos PUBLIC m)
endif()

if (OPENBUENOS_BUILD_SHARED)
  add_library(openbuenos_shared SHARED ${OPENBUENOS_SOURCES})
  set_target_properties(openbuenos_shared PROPERTIES OUTPUT_NAME openbuenos)
  target_compile_features(openbuenos_shared PRIVATE cxx_std_17)
  target_compile_options(openbuenos_shared PRIVATE
    -Wall -Wextra -Werror -fno-exceptions -fno-rtti
  )
  target_include_directories(openbuenos_shared
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${OPENBUENOS_ROOT}/docs/baremetal-vii
      ${OPENBUENOS_ROOT}/docs/kernel-abi
  )
  if (UNIX)
    target_link_libraries(openbuenos_shared PUBLIC m)
  endif()
endif()

if (OPENBUENOS_BUILD_TESTS)
  add_executable(openbuenos_selftest tests/openbuenos_selftest.cpp)
  target_link_libraries(openbuenos_selftest PRIVATE openbuenos)
  target_include_directories(openbuenos_selftest PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OPENBUENOS_ROOT}/docs/baremetal-vii
    ${OPENBUENOS_ROOT}/docs/kernel-abi
  )
  target_compile_features(openbuenos_selftest PRIVATE cxx_std_17)
  target_compile_options(openbuenos_selftest PRIVATE -Wall -Wextra -Werror -fno-exceptions -fno-rtti)
  if (ANDROID)
    target_compile_options(openbuenos_selftest PRIVATE -fPIE)
    target_link_options(openbuenos_selftest PRIVATE -pie -static-libstdc++ -Wl,--as-needed)
  endif()

  if (NOT ANDROID)
    add_executable(openbuenos_host_test tests/openbuenos_host_test.cpp)
    target_link_libraries(openbuenos_host_test PRIVATE openbuenos)
    target_include_directories(openbuenos_host_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${OPENBUENOS_ROOT}/docs/baremetal-vii
      ${OPENBUENOS_ROOT}/docs/kernel-abi
    )
    target_compile_features(openbuenos_host_test PRIVATE cxx_std_17)
    target_compile_options(openbuenos_host_test PRIVATE -Wall -Wextra -Werror -fno-exceptions -fno-rtti)
    enable_testing()
    add_test(NAME openbuenos_host_test COMMAND openbuenos_host_test)
  endif()
endif()
