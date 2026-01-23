#ifndef TOY_COMPILER_STRATEGY_H
#define TOY_COMPILER_STRATEGY_H

#include <mlir/IR/DialectInterface.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>

/*
 * Strategies interfaces for Toy dialect.
 * The implementations are placed under Device[X] directories.
 */

class ICombineStrategy {
public:
  virtual ~ICombineStrategy() = default;
  virtual void addPatterns(mlir::RewritePatternSet &patterns);
};

class IToyStrategy {
public:
  virtual ~IToyStrategy() = default;
  virtual std::unique_ptr<ICombineStrategy> getCombineStrategy();
};

class ToyStrategyCache final
    : public mlir::DialectInterface::Base<ToyStrategyCache> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyStrategyCache)

  explicit ToyStrategyCache(mlir::Dialect *dialect) : Base(dialect) {
  }

  std::unique_ptr<IToyStrategy> &getToyStrategy() { return _toyStrategy; }

  void setToyStrategy(std::unique_ptr<IToyStrategy> toyStrategy) {
    _toyStrategy = std::move(toyStrategy);
  }

private:
  std::unique_ptr<IToyStrategy> _toyStrategy = nullptr;
};

void setToyStrategy(mlir::MLIRContext *context, std::unique_ptr<IToyStrategy> strategy);
std::unique_ptr<IToyStrategy> &getToyStrategy(mlir::MLIRContext *context);

#endif // TOY_COMPILER_STRATEGY_H
