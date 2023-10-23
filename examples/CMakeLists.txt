include(HandleLLVMOptions)

add_llvm_executable(add-values add-values.cpp PARTIAL_SOURCES_INTENDED)
target_link_libraries(add-values
    PRIVATE
        MLIRCAPIConversion
        MLIRCAPIExecutionEngine
        MLIRCAPIRegisterEverything)
install(TARGETS add-values RUNTIME DESTINATION bin LIBRARY DESTINATION bin ARCHIVE DESTINATION lib)
