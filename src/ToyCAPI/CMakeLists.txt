add_mlir_public_c_api_library(ToyCAPI
        Dialects.cpp
    DISABLE_INSTALL
    LINK_LIBS PUBLIC
        ToyIR
        ToyTransforms)
