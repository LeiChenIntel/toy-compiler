#include "Conversion/Passes.h"
#include "Toy/Dialect.h"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/Pass/Pass.h>

using namespace mlir;
using namespace mlir::toy;

#define GEN_PASS_CLASSES
#include "Conversion/Passes.h.inc"

namespace {

class ConvertToyToMedian : public ConvertToyToMedianBase<ConvertToyToMedian> {
public:
  ConvertToyToMedian() = default;

  void runOnOperation() override {}
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::toy::createConvertToyToMedianPass() {
  return std::make_unique<::ConvertToyToMedian>();
}
