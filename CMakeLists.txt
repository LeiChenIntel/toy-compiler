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

# Use /MD instead of /MT in googletest
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory(thirdparty/googletest)

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(${PROJECT_BINARY_DIR}/include) # Include generated files

add_subdirectory(include/Toy)
add_subdirectory(src)

add_executable(unit_test mlir_unit_test.cpp)
target_include_directories(unit_test SYSTEM PUBLIC ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_link_libraries(unit_test PUBLIC MLIRIR)
target_link_libraries(unit_test PUBLIC gtest gtest_main)
install(TARGETS unit_test RUNTIME DESTINATION bin LIBRARY DESTINATION bin ARCHIVE DESTINATION lib)
