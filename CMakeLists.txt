cmake_minimum_required(VERSION 3.10)

project(Toy-Compiler)
message(STATUS "PROJECT_NAME: " ${PROJECT_NAME})

# LLVM/MLIR requires c++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Use /MD instead of /MT in googletest
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory(thirdparty/googletest)

find_package(MLIR REQUIRED CONFIG)

add_executable(unit_test mlir_unit_test.cpp)
target_include_directories(unit_test SYSTEM PUBLIC ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_link_libraries(unit_test PUBLIC MLIRIR)
target_link_libraries(unit_test PUBLIC gtest gtest_main)
