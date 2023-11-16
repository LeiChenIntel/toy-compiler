include(AddMLIRPython)

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_toy.")

declare_mlir_python_sources(ToyPythonSources)

declare_mlir_dialect_python_bindings(
    ADD_TO_PARENT ToyPythonSources
    ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
    TD_FILE PythonToyOps.td
    SOURCES toy.py
    DIALECT_NAME toy)

declare_mlir_python_extension(ToyPythonSources.Extension
    MODULE_NAME _toyDialects
    ADD_TO_PARENT ToyPythonSources
    SOURCES ToyExtension.cpp)
