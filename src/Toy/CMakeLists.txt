add_mlir_dialect_library(
    ToyIR
        Dialect.cpp
    DEPENDS
        ToyTableGen)
target_include_directories(ToyIR PUBLIC ${PROJECT_SOURCE_DIR}/include)
