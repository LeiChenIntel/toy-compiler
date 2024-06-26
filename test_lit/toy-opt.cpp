#include "Conversion/Passes.h"
#include "Toy/Dialect.h"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/InitAllExtensions.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

int main(int argc, char **argv) {
  mlir::registerCanonicalizerPass();
  mlir::toy::registerConvertToyToMid();
  mlir::toy::registerConvertMidToLLVM();

  mlir::DialectRegistry registry;
  mlir::func::registerAllExtensions(registry);

  mlir::MLIRContext context(registry);
  registry.insert<mlir::toy::ToyDialect, mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect, mlir::BuiltinDialect,
                  mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::LLVM::LLVMDialect,
                  mlir::vector::VectorDialect, mlir::amx::AMXDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Toy optimizer tool\n", registry));
}
