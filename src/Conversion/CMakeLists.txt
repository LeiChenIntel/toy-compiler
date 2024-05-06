add_mlir_conversion_library(
    ToyConversion
        ToyToMidPass.cpp
        MidToLLVMPass.cpp
    DEPENDS
    ToyOpsIncGen
    ToyInterfaceIncGen
    ToyConversionPassIncGen)
target_include_directories(ToyConversion PUBLIC ${PROJECT_SOURCE_DIR}/include)
