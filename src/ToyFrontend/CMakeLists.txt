add_library(ToyFrontend STATIC AST.cpp MLIRGen.cpp)
if(MSVC)
    # enable exceptions in vector.push_back
    target_compile_options(ToyFrontend PRIVATE /EHsc)
endif()
target_include_directories(ToyFrontend PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(ToyFrontend SYSTEM PUBLIC ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_link_libraries(ToyFrontend PRIVATE MLIRSupport MLIRIR ToyIR)
install(TARGETS ToyFrontend RUNTIME DESTINATION bin LIBRARY DESTINATION bin ARCHIVE DESTINATION lib)
