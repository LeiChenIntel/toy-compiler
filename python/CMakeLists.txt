include(AddMLIRPython)
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_toy.")

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(ToyPythonSources)

declare_mlir_dialect_python_bindings(
    ADD_TO_PARENT ToyPythonSources
    ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
    TD_FILE dialects/ToyOps.td
    SOURCES
        dialects/toy_pybind11.py
    DIALECT_NAME toy)
