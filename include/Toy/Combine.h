#ifndef TOY_COMPILER_COMBINE_H
#define TOY_COMPILER_COMBINE_H

#include "Toy/Dialect.h"

#include <mlir/IR/PatternMatch.h>

// Reshape(Constant(x)) = x'
struct FoldConstantReshapeOptPattern final
    : mlir::OpRewritePattern<mlir::toy::ReshapeOp> {
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  /// More like an order? benefit=1 is the first pattern to match and rewrite.
  explicit FoldConstantReshapeOptPattern(mlir::MLIRContext *context)
    : OpRewritePattern<mlir::toy::ReshapeOp>(context, /*benefit=*/1) {
  }

  mlir::LogicalResult matchAndRewrite(mlir::toy::ReshapeOp op, mlir::PatternRewriter &rewriter) const override;
};

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
#include "Toy/ToyCombine.inc"
} // namespace

#endif // TOY_COMPILER_COMBINE_H
