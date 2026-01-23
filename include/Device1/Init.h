#ifndef DEVICE1_INIT_H
#define DEVICE1_INIT_H

#include "Init/Registry.h"

class Device1Initializer final : public IStrategiesInitializer {
public:
  void initialize(mlir::MLIRContext *context) override;
};

#endif // DEVICE1_INIT_H
