#ifndef TOY_COMPILER_PASSES_H
#define TOY_COMPILER_PASSES_H

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

namespace mlir {
namespace toy {

/// Create a pass for lowering to operations in the `Affine` and `Std` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<OperationPass<ModuleOp>> createConvertToyToMidPass();

#define GEN_PASS_REGISTRATION
#include "Conversion/Passes.h.inc"

} // namespace toy
} // namespace mlir

#endif // TOY_COMPILER_PASSES_H
