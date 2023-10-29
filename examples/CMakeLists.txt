include(HandleLLVMOptions)

add_llvm_executable(add-values add-values.cpp PARTIAL_SOURCES_INTENDED)
target_link_libraries(add-values
    PRIVATE
        MLIRCAPIConversion
        MLIRCAPIExecutionEngine
        MLIRCAPIRegisterEverything)
install(TARGETS add-values RUNTIME DESTINATION bin LIBRARY DESTINATION bin ARCHIVE DESTINATION lib)

add_llvm_executable(print-tensor print-tensor.cpp PARTIAL_SOURCES_INTENDED)
target_link_libraries(print-tensor PRIVATE ToyIR ToyFrontend ToyTransforms ToyConversion)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(print-tensor PRIVATE ${dialect_libs} MLIROptLib MLIRTransforms MLIRExecutionEngine)
install(TARGETS print-tensor RUNTIME DESTINATION bin LIBRARY DESTINATION bin ARCHIVE DESTINATION lib)
