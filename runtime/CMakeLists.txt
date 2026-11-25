cmake_minimum_required(VERSION 3.18)
project(freetpu_runtime VERSION 0.1.0 LANGUAGES C CXX)

include(GNUInstallDirs)

option(FREETPU_RUNTIME_WERROR "Treat compiler warnings as errors" ON)

set(FREETPU_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

if (UNIX AND NOT APPLE)
  add_executable(tpu_baremetal_runner src/tpu_baremetal_runner.cpp)
  target_include_directories(tpu_baremetal_runner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${FREETPU_ROOT}/docs/baremetal-vii
    ${FREETPU_ROOT}/docs/kernel-abi
  )
  target_compile_features(tpu_baremetal_runner PRIVATE cxx_std_17)
  if (FREETPU_RUNTIME_WERROR)
    target_compile_options(tpu_baremetal_runner PRIVATE -Wall -Wextra -Werror)
  else()
    target_compile_options(tpu_baremetal_runner PRIVATE -Wall -Wextra)
  endif()
  if (ANDROID)
    target_compile_options(tpu_baremetal_runner PRIVATE -fPIE)
    target_link_options(tpu_baremetal_runner PRIVATE -pie -static-libstdc++ -Wl,--as-needed)
  else()
    install(TARGETS tpu_baremetal_runner RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
else()
  message(STATUS "tpu_baremetal_runner requires Linux or Android UAPI headers (<linux/ioctl.h>, <sys/eventfd.h>); skipping on ${CMAKE_SYSTEM_NAME}")
endif()
