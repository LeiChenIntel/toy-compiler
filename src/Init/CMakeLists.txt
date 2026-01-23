add_mlir_library(
    ToyInit
        Registry.cpp
    DISABLE_INSTALL
    LINK_LIBS
        ToyDevice1
        ToyDevice2)
target_include_directories(ToyInit PUBLIC ${PROJECT_SOURCE_DIR}/include)
