#include "Device2/Init.h"
#include "Toy/Strategies.h"
#include "Toy/Combine.h"

class CombineStrategyDevice2 final : public ICombineStrategy {
  void addPatterns(mlir::RewritePatternSet &patterns) override {
    auto ctx = patterns.getContext();
    patterns.add<ReshapeReshapeOptPattern, RedundantReshapeOptPattern>(ctx);
  }
};

class ToyStrategyDevice2 final : public IToyStrategy {
  std::unique_ptr<ICombineStrategy> getCombineStrategy() override {
    return std::make_unique<CombineStrategyDevice2>();
  }
};

void Device2Initializer::initialize(mlir::MLIRContext *context) {
  auto strategy = std::make_unique<ToyStrategyDevice2>();
  setToyStrategy(context, std::move(strategy));
}
