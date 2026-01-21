#include "Toy/Dialect.h"
#include "Toy/Combine.h"
#include "Toy/Strategies.h"

#include <mlir/IR/PatternMatch.h>

using namespace mlir;
using namespace toy;

mlir::LogicalResult FoldConstantReshapeOptPattern::matchAndRewrite(mlir::toy::ReshapeOp op,
                                                                   mlir::PatternRewriter &rewriter) const {
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

/// Register our patterns as "canonicalization" patterns on the ReshapeOp so
/// that they can be picked up by the Canonicalization framework.
void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  const auto &strategies = getToyStrategy(context);
  const auto combineStrategy = strategies->getCombineStrategy();
  combineStrategy->addPatterns(results);
}
