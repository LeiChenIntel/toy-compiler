add_mlir_library(
        Init
        Registry.cpp
        DISABLE_INSTALL)
target_include_directories(Init PUBLIC ${PROJECT_SOURCE_DIR}/include)
