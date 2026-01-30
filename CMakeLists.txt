cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0116 NEW) # Avoid Ninja generators transform DEPFILEs CMake warning
cmake_policy(SET CMP0077 NEW) # Avoid warning when adds HandleLLVMOptions

project(Toy-Compiler)
message(STATUS "PROJECT_NAME: " ${PROJECT_NAME})

# LLVM/MLIR requires c++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# set install path
set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}")

# Define options
# Need AVX2 support for benchmarks! Only support release mode!
option(ENABLE_TOY_BENCHMARKS "" ON)
if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(WARNING "Please set ENABLE_TOY_BENCHMARKS OFF for RelWithDebInfo and Debug.")
endif()
# Need OpenBLAS library for matrix multiplication benchmarks!
option(ENABLE_MATMUL_BENCHMARKS "" OFF)
# Recommend to set up virtual environment for Python bindings
option(ENABLE_PYTHON_BINDINGS "" OFF)

# Use /MD instead of /MT in googletest
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF)
add_subdirectory(thirdparty/googletest)

# Need to find Python package to enable LIT on Windows platform
if(MSVC)
    find_package(Python3 ${LLVM_MINIMUM_PYTHON_VERSION} REQUIRED COMPONENTS Interpreter)
endif()
find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
if(ENABLE_PYTHON_BINDINGS)
    include(MLIRDetectPythonEnv)
    mlir_configure_python_dev_packages()
endif()

include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include) # Include .td files
include_directories(${PROJECT_BINARY_DIR}/include) # Include generated files

# Add all targets depends on all-targets
# Install should be triggered after all targets built
add_custom_target(all-targets ALL)
add_custom_command(
    TARGET all-targets
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} --install .
    COMMENT "Install project ...")

add_subdirectory(examples)
add_subdirectory(include/Conversion)
add_subdirectory(include/Toy)
add_subdirectory(src)
add_subdirectory(test_lit)
add_subdirectory(test_unit)

if(ENABLE_PYTHON_BINDINGS)
    message(STATUS "Enable Python bindings")
    add_subdirectory(python)
endif()

if(ENABLE_TOY_BENCHMARKS)
    message(STATUS "Enable Toy benchmarks")
    add_subdirectory(benchmarks)
endif()
