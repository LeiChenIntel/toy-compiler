#include "Toy/Strategies.h"
#include "Toy/Dialect.h"

template <typename Cache, typename Dialect>
Cache &getCache(mlir::MLIRContext *ctx) {
  auto *dialect = ctx->getOrLoadDialect<Dialect>();
  assert(dialect != nullptr && "dialect must be present in the context");
  auto *iface = dialect->template getRegisteredInterface<Cache>();
  // addInterface to dialect if failed, example: addInterfaces<ToyStrategyCache>()
  assert(iface != nullptr && "The requested cache must be registered in the context");
  return *iface;
}

void setToyStrategy(mlir::MLIRContext *context, std::unique_ptr<IToyStrategy> strategy) {
  auto &registeredInterface = getCache<ToyStrategyCache, mlir::toy::ToyDialect>(context);
  registeredInterface.setToyStrategy(std::move(strategy));
}

std::unique_ptr<IToyStrategy> &getToyStrategy(mlir::MLIRContext *context) {
  auto &registeredInterface = getCache<ToyStrategyCache, mlir::toy::ToyDialect>(context);
  return registeredInterface.getToyStrategy();
}
