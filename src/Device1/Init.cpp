#include "Device1/Init.h"
#include "Toy/Strategies.h"
#include "Toy/Combine.h"

class CombineStrategyDevice1 final : public ICombineStrategy {
  void addPatterns(mlir::RewritePatternSet &patterns) override {
    auto ctx = patterns.getContext();
    patterns.add<ReshapeReshapeOptPattern, RedundantReshapeOptPattern,
                 FoldConstantReshapeOptPattern>(ctx);
  }
};

class ToyStrategyDevice1 final : public IToyStrategy {
  std::unique_ptr<ICombineStrategy> getCombineStrategy() override {
    return std::make_unique<CombineStrategyDevice1>();
  }
};

void Device1Initializer::initialize(mlir::MLIRContext *context) {
  auto strategy = std::make_unique<ToyStrategyDevice1>();
  setToyStrategy(context, std::move(strategy));
}
