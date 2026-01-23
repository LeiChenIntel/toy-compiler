include(HandleLLVMOptions)

add_llvm_executable(toy-opt toy-opt.cpp PARTIAL_SOURCES_INTENDED)
# toy-opt is designed to test single pass lit
target_include_directories(toy-opt SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
target_link_libraries(toy-opt PRIVATE ${dialect_libs} ${extension_libs} MLIROptLib MLIRTransforms MLIRPass
        MLIRVectorToLLVM)
target_link_libraries(toy-opt PRIVATE ToyIR ToyTransforms ToyConversion ToyInit ToyDevice1 Utils)
add_dependencies(all-targets toy-opt)
install(TARGETS toy-opt RUNTIME DESTINATION bin LIBRARY DESTINATION bin ARCHIVE DESTINATION lib)

# toy-translate is designed to translate one IR to another IR
# basically, toy-translate is a pass pipeline and may include many passes
add_llvm_executable(toy-translate toy-translate.cpp PARTIAL_SOURCES_INTENDED)
target_include_directories(toy-translate PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(toy-translate SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_link_libraries(toy-translate PRIVATE ${dialect_libs} ${extension_libs} MLIRSupport MLIRIR MLIRTransforms
        MLIRInferTypeOpInterface MLIRExecutionEngine)
target_link_libraries(toy-translate PRIVATE ToyIR ToyFrontend ToyTransforms ToyConversion ToyInit ToyDevice1 Utils)
add_dependencies(all-targets toy-translate)
install(TARGETS toy-translate RUNTIME DESTINATION bin LIBRARY DESTINATION bin ARCHIVE DESTINATION lib)

configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

add_lit_testsuite(check-toy-lit
    "Running the toy lit tests"
    ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS
        FileCheck
        toy-opt
        toy-translate)
