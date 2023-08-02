include(HandleLLVMOptions)

add_subdirectory(Toy)

add_llvm_executable(toy_compiler ToyCompiler.cpp AST.cpp MLIRGen.cpp)
target_include_directories(toy_compiler PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(toy_compiler SYSTEM PUBLIC ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_link_libraries(toy_compiler PRIVATE MLIRSupport MLIRIR ToyIR)
install(TARGETS toy_compiler RUNTIME DESTINATION bin LIBRARY DESTINATION bin ARCHIVE DESTINATION lib)
