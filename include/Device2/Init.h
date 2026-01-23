#ifndef DEVICE2_INIT_H
#define DEVICE2_INIT_H

#include "Init/Registry.h"

class Device2Initializer final : public IStrategiesInitializer {
public:
  void initialize(mlir::MLIRContext *context) override;
};

#endif // DEVICE2_INIT_H
