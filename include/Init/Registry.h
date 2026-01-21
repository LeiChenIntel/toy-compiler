#ifndef INIT_REGISTRY_H
#define INIT_REGISTRY_H

#include <mlir/IR/DialectRegistry.h>

class IStrategiesInitializer {
public:
  virtual void initialize(mlir::MLIRContext *context) = 0;
  virtual ~IStrategiesInitializer() = default;
};

enum class Platform { Device1 = 0, Device2 };

namespace toy {
void registerToyStrategies(mlir::DialectRegistry &registry, Platform device);
}

#endif // INIT_REGISTRY_H
