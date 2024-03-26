#ifndef TOY_COMPILER_PASSES_H
#define TOY_COMPILER_PASSES_H

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

#include "Toy/Dialect.h"

namespace mlir {

//
// Common functions
//

/// Convert the given TensorType into the corresponding MemRefType.
static MemRefType convertTensorToMemRef(TensorType type) {
  assert(type.hasRank() && "expected only ranked shapes");
  return MemRefType::get(type.getShape(), type.getElementType());
}

/// Insert an allocation and deallocation for the given MemRefType.
static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter) {
  auto alloc = rewriter.create<memref::AllocOp>(loc, type);

  // Make sure to allocate at the beginning of the block.
  auto *parentBlock = alloc->getBlock();
  alloc->moveBefore(&parentBlock->front());

  // Make sure to deallocate this alloc at the end of the block. This is fine
  // as toy functions have no control flow.
  auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
  dealloc->moveBefore(&parentBlock->back());
  return alloc;
}

static Value createStoreOpMemRef(Operation *op, PatternRewriter &rewriter) {
  const auto tensorType = (*op->result_type_begin()).cast<TensorType>();
  const auto loc = op->getLoc();
  const auto opResVal = op->getResult(0);
  mlir::Value memRef;
  // Need to check if the result is stored to input pointers.
  bool isMemAllocated = false;

  // Find if the store operation exists.
  auto &blk = op->getParentRegion()->front();
  const auto storeOps = blk.getOps<toy::StoreOp>();
  for (auto op : storeOps) {
    // If the operation result is stored, return the buffer
    if (opResVal == op.getValToStore()) {
      memRef = op.getMemref();
      isMemAllocated = true;
      rewriter.eraseOp(op);
      break;
    }
  }

  if (!isMemAllocated) {
    // Insert an allocation and deallocation for the result of this operation.
    const auto memRefType = convertTensorToMemRef(tensorType);
    memRef = insertAllocAndDealloc(memRefType, loc, rewriter);
  }

  return memRef;
}
} // namespace mlir

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

std::unique_ptr<OperationPass<ModuleOp>> createConvertToySubToMidPass();

#define GEN_PASS_REGISTRATION
#include "Conversion/Passes.h.inc"

} // namespace toy
} // namespace mlir

#endif // TOY_COMPILER_PASSES_H
