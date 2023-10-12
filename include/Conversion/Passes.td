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

def ConvertMidToLLVM: Pass<"convert-mid-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert Mid to LLVM Operations";
    let description = [{

    }];
    let constructor = "mlir::toy::createConvertMidToLLVMPass()";

    let dependentDialects = [
      "mlir::LLVM::LLVMDialect",
      "mlir::scf::SCFDialect"];
}

#endif // CONVERSION_PASSES_TD
