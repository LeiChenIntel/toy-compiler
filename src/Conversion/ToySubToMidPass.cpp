#include "Conversion/Passes.h"
#include "Toy/Dialect.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "Conversion/Passes.h.inc"

namespace {
class ConvertToySubToMid : public ConvertToySubToMidBase<ConvertToySubToMid> {
public:
  ConvertToySubToMid() = default;
  void runOnOperation() override {
    ConversionTarget target(getContext());
    // target.addLegalDialect<AffineDialect, BuiltinDialect, arith::ArithDialect,
    //                        func::FuncDialect>();
    target.addIllegalOp<toy::SubOp>();

    RewritePatternSet patterns(&getContext());

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
