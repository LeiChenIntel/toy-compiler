#include "Init/Registry.h"
#include "Toy/Dialect.h"

#include <mlir/IR/DialectRegistry.h>

std::unique_ptr<IStrategiesInitializer> createStrategiesInitializer(const Platform device) {
  switch (device) {
  case Platform::Device1:
    return nullptr;
  case Platform::Device2:
    return nullptr;
  default:
    std::runtime_error("Unknown device");
    return nullptr;
  }
}

class StrategiesExtension final
    : public mlir::DialectExtension<StrategiesExtension, mlir::toy::ToyDialect> {
public:
  explicit StrategiesExtension(const Platform device) : _device(device) {
  }

  void apply(mlir::MLIRContext *context, mlir::toy::ToyDialect *) const override {
    const auto strategiesInitializer = createStrategiesInitializer(_device);
    // Initialize the strategies of one device in the context
    strategiesInitializer->initialize(context);
  }

private:
  Platform _device;
};

void registerToyStrategies(mlir::DialectRegistry &registry, Platform device) {
  registry.addExtension(mlir::TypeID::get<StrategiesExtension>(), std::make_unique<StrategiesExtension>(device));
}
