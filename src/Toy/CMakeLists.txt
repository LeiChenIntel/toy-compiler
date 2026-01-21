add_mlir_dialect_library(
    ToyIR
        Dialect.cpp
        Ops.cpp
        Strategies.cpp
    PARTIAL_SOURCES_INTENDED
    DEPENDS
        ToyOpsIncGen)
target_include_directories(ToyIR PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_mlir_dialect_library(
    ToyTransforms
        Combine.cpp
    PARTIAL_SOURCES_INTENDED
    DEPENDS
        ToyOpsIncGen
        ToyCombineIncGen)
