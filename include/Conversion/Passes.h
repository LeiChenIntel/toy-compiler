#ifndef TOY_COMPILER_PASSES_H
#define TOY_COMPILER_PASSES_H

#include <mlir/Dialect/AMX/AMXDialect.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir {
namespace toy {

enum LoweringPatternMode { Loop = 0, Vector };

/// Create a pass for lowering to operations in the `Affine` and `Std` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<OperationPass<ModuleOp>> createConvertToyToMidPass();

std::unique_ptr<OperationPass<ModuleOp>>
createConvertToyToMidPass(LoweringPatternMode mode);

/// Create a pass for lowering operations the remaining `Toy` operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<OperationPass<ModuleOp>> createConvertMidToLLVMPass();

#define GEN_PASS_REGISTRATION
#include "Conversion/Passes.h.inc"

} // namespace toy
} // namespace mlir

#endif // TOY_COMPILER_PASSES_H
