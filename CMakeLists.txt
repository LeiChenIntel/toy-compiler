cmake_minimum_required(VERSION 3.10)

project(Toy-Compiler)
message(STATUS "PROJECT_NAME: " ${PROJECT_NAME})

# LLVM/MLIR requires c++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# set install path
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}")
endif ()

# Use /MD instead of /MT in googletest
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory(thirdparty/googletest)

find_package(MLIR REQUIRED CONFIG)

add_subdirectory(src)

add_executable(unit_test mlir_unit_test.cpp)
target_include_directories(unit_test SYSTEM PUBLIC ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_link_libraries(unit_test PUBLIC MLIRIR)
target_link_libraries(unit_test PUBLIC gtest gtest_main)
install(TARGETS unit_test RUNTIME DESTINATION bin LIBRARY DESTINATION bin ARCHIVE DESTINATION lib)
