add_mlir_library(
    ToyInit
        Registry.cpp
    DISABLE_INSTALL)
target_include_directories(ToyInit PUBLIC ${PROJECT_SOURCE_DIR}/include)
