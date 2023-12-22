# Generate libraries from Toy
if(MSVC)
    set(TOY_TRANSLATE_EXE ${PROJECT_BINARY_DIR}/test_lit/Release/toy-translate.exe)
    set(LLVM_LLC_EXE ${LLVM_BINARY_DIR}/Release/bin/llc.exe)
    set(TRIPLE x86_64-pc-windows-msvc)
elseif(UNIX)
    set(TOY_TRANSLATE_EXE ${PROJECT_BINARY_DIR}/test_lit/toy-translate)
    set(LLVM_LLC_EXE ${LLVM_BINARY_DIR}/bin/llc)
    set(TRIPLE x86_64-unknown-linux-gnu)
else()
    message(FATAL_ERROR "Unsupported operating system")
endif()
#add_custom_command(OUTPUT add_tensors_8192.o
#    COMMAND ${TOY_TRANSLATE_EXE} ${CMAKE_CURRENT_SOURCE_DIR}/add_tensors_8192.toy -emit=llvm -lower-pat=vector > add_tensors_8192.mlir
#    COMMAND ${LLVM_LLC_EXE} add_tensors_8192.mlir -mtriple=${TRIPLE} -mattr=avx2 --filetype=obj -o add_tensors_8192.o
#    DEPENDS toy-translate)

add_library(AddTensors8192 STATIC add_tensors_8192.o)
set_target_properties(AddTensors8192 PROPERTIES LINKER_LANGUAGE CXX)

add_executable(benchmark-add-tensors benchmark-add-tensors.cpp)
add_dependencies(benchmark-add-tensors toy-translate)
if(MSVC)
    target_compile_options(benchmark-add-tensors PUBLIC "/arch:AVX2")
elseif(UNIX)
    target_compile_options(benchmark-add-tensors PUBLIC -mavx2)
endif()
target_link_libraries(benchmark-add-tensors PRIVATE AddTensors8192)
install(TARGETS benchmark-add-tensors RUNTIME DESTINATION bin LIBRARY DESTINATION bin ARCHIVE DESTINATION lib)
