include(AddMLIRPython)
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_toy.")

################################################################################
# Sources, prepare source and define modules for Toy dialect Python bindings.
################################################################################

declare_mlir_python_sources(ToyPythonSources)

declare_mlir_dialect_python_bindings(
    ADD_TO_PARENT ToyPythonSources
    ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
    TD_FILE dialects/ToyOps.td
    SOURCES
        dialects/toy_pybind11.py
    DIALECT_NAME toy)

declare_mlir_python_extension(ToyPythonSources.Pybind11Extension
    MODULE_NAME _toyDialectsPybind11
    ADD_TO_PARENT ToyPythonSources
    SOURCES
        ToyExtensionPybind11.cpp
    EMBED_CAPI_LINK_LIBS
        ToyCAPI
    PYTHON_BINDINGS_LIBRARY pybind11)

################################################################################
# Common CAPI, create library using sources defined above.
# ToyPythonCAPI will not be created unless it is linked by some target!
################################################################################

add_mlir_python_common_capi_library(ToyPythonCAPI
    INSTALL_COMPONENT ToyPythonModules
    INSTALL_DESTINATION python_packages/toy/mlir_toy/_mlir_libs
    OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python_packages/toy/mlir_toy/_mlir_libs"
    #RELATIVE_INSTALL_ROOT "../../../.."
    DECLARED_SOURCES
        ToyPythonSources
        MLIRPythonExtension.RegisterEverything
        MLIRPythonSources.Core)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(ToyPythonModules
    ROOT_PREFIX "${CMAKE_BINARY_DIR}/python_packages/toy/mlir_toy"
    INSTALL_PREFIX "python_packages/toy/mlir_toy"
    DECLARED_SOURCES
        ToyPythonSources
        MLIRPythonExtension.RegisterEverything
        MLIRPythonSources
    COMMON_CAPI_LINK_LIBS
        ToyPythonCAPI)
