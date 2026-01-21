add_mlir_library(
    ToyDevice1
        Init.cpp
    DISABLE_INSTALL
    LINK_LIBS
        ToyIR)
target_include_directories(ToyDevice1 PUBLIC ${PROJECT_SOURCE_DIR}/include)
