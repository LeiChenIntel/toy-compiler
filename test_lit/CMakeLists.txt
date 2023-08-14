include(HandleLLVMOptions)

add_llvm_executable(toy-opt toy-opt.cpp PARTIAL_SOURCES_INTENDED)
# toy-opt is designed to test single pass lit
target_include_directories(toy-opt SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_link_libraries(toy-opt PRIVATE MLIROptLib MLIRTransforms MLIRPass)
target_link_libraries(toy-opt PRIVATE ToyIR ToyTransforms)
install(TARGETS toy-opt RUNTIME DESTINATION bin LIBRARY DESTINATION bin ARCHIVE DESTINATION lib)

# toy-translate is designed to translate one IR to another IR
# basically, toy-translate is a pass pipeline and may include many passes
add_llvm_executable(toy-translate toy-translate.cpp PARTIAL_SOURCES_INTENDED)
target_include_directories(toy-translate PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(toy-translate SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_link_libraries(toy-translate PRIVATE MLIRSupport MLIRIR ToyIR ToyFrontend ToyTransforms)
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
