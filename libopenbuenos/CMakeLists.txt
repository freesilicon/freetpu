cmake_minimum_required(VERSION 3.18)
project(openbuenos VERSION 0.1.0 LANGUAGES C CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(OPENBUENOS_BUILD_TESTS "Build libopenbuenos self-test / host unit test binaries" ON)
option(OPENBUENOS_INSTALL_SELFTEST "Install openbuenos_selftest utility" ON)
option(OPENBUENOS_WERROR "Treat compiler warnings as errors" ON)

set(OPENBUENOS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(OPENBUENOS_COMPILE_FLAGS -Wall -Wextra -fno-exceptions -fno-rtti)
if (OPENBUENOS_WERROR)
  list(APPEND OPENBUENOS_COMPILE_FLAGS -Werror)
endif()

add_library(openbuenos STATIC
  src/ob_sha256.cpp
  src/ob_dgc0.cpp
  src/ob_buffer_pool.cpp
  src/ob_worker_pool.cpp
  src/ob_device.cpp
  src/ob_graph.cpp
  src/ob_gemm_emitter.cpp
  src/ob_gemm.cpp
)

add_library(openbuenos::openbuenos ALIAS openbuenos)

set_target_properties(openbuenos PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(openbuenos PUBLIC cxx_std_17)
target_compile_options(openbuenos PRIVATE ${OPENBUENOS_COMPILE_FLAGS})

target_include_directories(openbuenos
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${OPENBUENOS_ROOT}/docs/baremetal-vii>
    $<BUILD_INTERFACE:${OPENBUENOS_ROOT}/docs/kernel-abi>
)

if (UNIX AND NOT APPLE)
  target_link_libraries(openbuenos PUBLIC m)
  if (NOT ANDROID)
    find_package(Threads REQUIRED)
    target_link_libraries(openbuenos PUBLIC Threads::Threads)
  endif()
elseif (APPLE)
  target_link_libraries(openbuenos PUBLIC m)
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
  target_compile_options(openbuenos_selftest PRIVATE ${OPENBUENOS_COMPILE_FLAGS})
  if (ANDROID)
    target_compile_options(openbuenos_selftest PRIVATE -fPIE)
    target_link_options(openbuenos_selftest PRIVATE -pie -static-libstdc++ -Wl,--as-needed)
  elseif (UNIX AND NOT APPLE AND OPENBUENOS_INSTALL_SELFTEST)
    install(TARGETS openbuenos_selftest RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()

  add_executable(tpu_soc_probe tests/tpu_soc_probe.cpp)
  target_link_libraries(tpu_soc_probe PRIVATE openbuenos)
  target_include_directories(tpu_soc_probe PRIVATE
    ${OPENBUENOS_ROOT}/docs/kernel-abi
  )
  target_compile_features(tpu_soc_probe PRIVATE cxx_std_17)
  target_compile_options(tpu_soc_probe PRIVATE ${OPENBUENOS_COMPILE_FLAGS})
  if (ANDROID)
    target_compile_options(tpu_soc_probe PRIVATE -fPIE)
    target_link_options(tpu_soc_probe PRIVATE -pie -static-libstdc++ -Wl,--as-needed)
  elseif (UNIX AND NOT APPLE AND OPENBUENOS_INSTALL_SELFTEST)
    install(TARGETS tpu_soc_probe RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
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
    target_compile_options(openbuenos_host_test PRIVATE ${OPENBUENOS_COMPILE_FLAGS})
    enable_testing()
    add_test(NAME openbuenos_host_test COMMAND openbuenos_host_test)
    add_test(NAME tpu_soc_probe_mock COMMAND tpu_soc_probe --mock)
  endif()
endif()

install(TARGETS openbuenos
  EXPORT openbuenosTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(FILES include/openbuenos.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(FILES ${OPENBUENOS_ROOT}/docs/kernel-abi/edgetpu_buenos_uapi.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/freetpu
)
install(EXPORT openbuenosTargets
  FILE openbuenosTargets.cmake
  NAMESPACE openbuenos::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/openbuenos
)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/openbuenosConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/openbuenosConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/openbuenos
)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/openbuenosConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/openbuenosConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/openbuenosConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/openbuenos
)
