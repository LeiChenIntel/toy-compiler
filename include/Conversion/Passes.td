#ifndef CONVERSION_PASSES_TD
#define CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertToyToMid: Pass<"convert-toy-to-mid", "mlir::ModuleOp"> {
    let summary = "Convert Toy to Mid Operations";
    let description = [{

    }];
    let constructor = "mlir::toy::createConvertToyToMidPass()";
    let options = [
      Option<"loweringPatternMode", "lowering-pattern-mode", "unsigned",
          /*default=loop*/"0", "Toy to Mid lowering pattern mode, 0 to loop, 1 to vector">
    ];

    let dependentDialects = [
      "mlir::affine::AffineDialect",
      "mlir::BuiltinDialect",
      "mlir::arith::ArithDialect",
      "mlir::func::FuncDialect",
      "mlir::memref::MemRefDialect",
      "mlir::vector::VectorDialect",
      "mlir::amx::AMXDialect",
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
