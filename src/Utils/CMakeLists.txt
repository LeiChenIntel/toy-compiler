add_mlir_library(
    Utils
        Utility.cpp
    DISABLE_INSTALL)
target_include_directories(Utils PUBLIC ${PROJECT_SOURCE_DIR}/include)
