#ifndef CONVERSION_PASSES_TD
#define CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertToyToMedian: Pass<"convert-toy-to-median", "mlir::ModuleOp"> {
    let summary = "Convert Toy to Median Operations";
    let description = [{

    }];
    let constructor = "mlir::toy::createConvertToyToMedianPass()";

    let dependentDialects = [
      "mlir::AffineDialect",
      "mlir::BuiltinDialect",
      "mlir::arith::ArithDialect",
      "mlir::func::FuncDialect",
      "mlir::memref::MemRefDialect",
      "mlir::toy::ToyDialect"];
}

#endif // CONVERSION_PASSES_TD
