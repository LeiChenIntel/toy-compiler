include(HandleLLVMOptions)

add_llvm_executable(add-values add-values.cpp PARTIAL_SOURCES_INTENDED)
target_link_libraries(add-values
    PRIVATE
        MLIRCAPIConversion
        MLIRCAPIExecutionEngine
        MLIRCAPIRegisterEverything)
add_dependencies(all-targets add-values)
install(TARGETS add-values RUNTIME DESTINATION bin LIBRARY DESTINATION bin ARCHIVE DESTINATION lib)

add_llvm_executable(add-tensors add-tensors.cpp PARTIAL_SOURCES_INTENDED)
target_link_libraries(add-tensors PRIVATE ToyIR ToyFrontend ToyTransforms ToyConversion)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
target_link_libraries(add-tensors PRIVATE ${dialect_libs} ${extension_libs} MLIROptLib MLIRTransforms MLIRExecutionEngine)
add_dependencies(all-targets add-tensors)
install(TARGETS add-tensors RUNTIME DESTINATION bin LIBRARY DESTINATION bin ARCHIVE DESTINATION lib)
