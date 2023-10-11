add_mlir_conversion_library(
    ToyConversion
        Passes.cpp
    DEPENDS
    ToyOpsIncGen
    ToyConversionPassIncGen)
target_include_directories(ToyConversion PUBLIC ${PROJECT_SOURCE_DIR}/include)
