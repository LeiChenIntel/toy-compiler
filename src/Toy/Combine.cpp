#include "Toy/Dialect.h"
#include <mlir/IR/PatternMatch.h>

using namespace mlir;
using namespace toy;

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "Toy/ToyCombine.inc"
} // namespace

// Reshape(Constant(x)) = x'
struct FoldConstantReshapeOptPattern
    : public mlir::OpRewritePattern<ReshapeOp> {
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  /// More like an order? benefit=1 is the first pattern to match and rewrite.
  explicit FoldConstantReshapeOptPattern(mlir::MLIRContext *context)
      : OpRewritePattern<ReshapeOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(ReshapeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto input = op.getOperand();
    ConstantOp cstInput = input.getDefiningOp<ConstantOp>();

    if (!cstInput) {
      return failure();
    }

    auto inputType = llvm::dyn_cast<TensorType>(input.getType());
    auto inputNumElements = inputType.getNumElements();
    auto inputEleType = inputType.getElementType();
    auto outputType = llvm::dyn_cast<TensorType>(op.getResult().getType());
    auto outputNumElements = outputType.getNumElements();
    auto outputEleType = outputType.getElementType();

    if (inputEleType != outputEleType) {
      // Reshape should not support type conversion operation
      return failure();
    }

    if (inputNumElements == outputNumElements) {
      // Reconstruct a ConstantOp with the result shape
      auto cstAttr = cstInput.getValueAttr().reshape(outputType);
      auto cstNew =
          rewriter.create<ConstantOp>(op->getLoc(), outputType, cstAttr);
      rewriter.replaceOp(op, cstNew.getResult());
      return success();
    }

    if (inputNumElements == 1 && inputNumElements != outputNumElements) {
      // Need to extend shape when there is a scalar input
      double data = cstInput.getValue().getSplatValue<double>();
      auto cstAttr =
          mlir::DenseElementsAttr::get(outputType, llvm::ArrayRef(data));
      auto cstNew =
          rewriter.create<ConstantOp>(op->getLoc(), outputType, cstAttr);
      rewriter.replaceOp(op, cstNew.getResult());
      return success();
    }

    return success();
  }
};

/// Register our patterns as "canonicalization" patterns on the ReshapeOp so
/// that they can be picked up by the Canonicalization framework.
void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<ReshapeReshapeOptPattern, RedundantReshapeOptPattern,
              FoldConstantReshapeOptPattern>(context);
}
