add_mlir_library(
    ToyDevice2
        Init.cpp
    DISABLE_INSTALL
    LINK_LIBS
        ToyIR)
target_include_directories(ToyDevice2 PUBLIC ${PROJECT_SOURCE_DIR}/include)
