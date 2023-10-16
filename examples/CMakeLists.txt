include(HandleLLVMOptions)

add_llvm_executable(add-tensor add-tensor.c PARTIAL_SOURCES_INTENDED)
target_link_libraries(add-tensor PRIVATE)
install(TARGETS add-tensor RUNTIME DESTINATION bin LIBRARY DESTINATION bin ARCHIVE DESTINATION lib)
