#include "Conversion/Passes.h"
#include "Toy/Dialect.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "Conversion/Passes.h.inc"

namespace {
class ToySubOpPattern : public ConversionPattern {
  // using OpConversionPattern<toy::SubOp>::OpConversionPattern;
public:
  ToySubOpPattern(MLIRContext *ctx)
      : ConversionPattern(toy::SubOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    // const auto tensorType = (op.getResult().getType().cast<TensorType>());
    const auto tensorType = (*op->result_type_begin()).cast<TensorType>();

    mlir::Value memRef = createStoreOpMemRef(op, rewriter);

    SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value=*/0);
    SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);
    const auto upperBounds = tensorType.getShape();
    ValueRange opes = operands;
    typename toy::SubOp::Adaptor adaptor(opes);
    buildAffineLoopNest(
        rewriter, loc, lowerBounds, upperBounds, steps,
        [&](OpBuilder &builder, Location loc, ValueRange loopIvs) {
          llvm::errs() << "check point 1\n";
          // typename toy::SubOp::Adaptor adaptor(op);
          auto loadedLhs =
              builder.create<AffineLoadOp>(loc, adaptor.getLhs(), loopIvs);
          llvm::errs() << "check point 2\n";
          auto loadedRhs =
              builder.create<AffineLoadOp>(loc, adaptor.getRhs(), loopIvs);
          llvm::errs() << "check point 3\n";
          Value valueToStore =
              rewriter.create<arith::SubFOp>(loc, loadedLhs, loadedRhs);
          builder.create<AffineStoreOp>(loc, valueToStore, memRef, loopIvs);
          llvm::errs() << "check point 4\n";
        });

    // replace op with arg2.
    // If op is used later, use values in arg2
    rewriter.replaceOp(op, memRef);
    return success();
  }
};

class ConvertToySubToMid : public ConvertToySubToMidBase<ConvertToySubToMid> {
public:
  ConvertToySubToMid() = default;
  void runOnOperation() override {
    ConversionTarget target(getContext());
    // target.addLegalDialect<AffineDialect, BuiltinDialect,
    // arith::ArithDialect,
    //                        func::FuncDialect>();
    target.addIllegalOp<toy::SubOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<ToySubOpPattern>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  };
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::toy::createConvertToySubToMidPass() {
  return std::make_unique<ConvertToySubToMid>();
}
