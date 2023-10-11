#ifndef CONVERSION_PASSES_TD
#define CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertToyToMid: Pass<"convert-toy-to-mid", "mlir::ModuleOp"> {
    let summary = "Convert Toy to Mid Operations";
    let description = [{

    }];
    let constructor = "mlir::toy::createConvertToyToMidPass()";

    let dependentDialects = [
      "mlir::AffineDialect",
      "mlir::BuiltinDialect",
      "mlir::arith::ArithDialect",
      "mlir::func::FuncDialect",
      "mlir::memref::MemRefDialect",
      "mlir::toy::ToyDialect"];
}

#endif // CONVERSION_PASSES_TD
