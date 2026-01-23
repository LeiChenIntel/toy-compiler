#ifndef INIT_REGISTRY_H
#define INIT_REGISTRY_H

#include "Utils/Utility.h"

#include <mlir/IR/DialectRegistry.h>

class IStrategiesInitializer {
public:
  virtual void initialize(mlir::MLIRContext *context) = 0;
  virtual ~IStrategiesInitializer() = default;
};

namespace toy {
void registerToyStrategies(mlir::DialectRegistry &registry, Platform device);
}

#endif // INIT_REGISTRY_H
