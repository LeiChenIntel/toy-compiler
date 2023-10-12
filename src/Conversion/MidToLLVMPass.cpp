#include "Conversion/Passes.h"
#include "Toy/Dialect.h"

#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>

using namespace mlir;

#define GEN_PASS_CLASSES
#include "Conversion/Passes.h.inc"

namespace {

class ConvertMidToLLVM : public ConvertMidToLLVMBase<ConvertMidToLLVM> {
public:
  ConvertMidToLLVM() = default;

  void runOnOperation() override {
    LLVMConversionTarget target(getContext());
    // Declare ModuleOp legal operation for FullConversion
    target.addLegalOp<mlir::ModuleOp>();
    // target.addLegalOp<mlir::func::FuncOp>();

    LLVMTypeConverter typeConverter(&getContext());

    RewritePatternSet patterns(&getContext());
    mlir::populateAffineToStdConversionPatterns(patterns);
    mlir::populateSCFToControlFlowConversionPatterns(patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          patterns);
    mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);

    // We want to completely lower to LLVM, so we use a `FullConversion`. This
    // ensures that only legal operations will remain after the conversion.
    auto moduleOp = getOperation();
    if (failed(applyFullConversion(moduleOp, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::toy::createConvertMidToLLVMPass() {
  return std::make_unique<ConvertMidToLLVM>();
}
