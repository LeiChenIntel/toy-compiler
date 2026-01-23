#include "Init/Registry.h"
#include "Toy/Dialect.h"
#include "Device1/Init.h"

#include <mlir/IR/DialectRegistry.h>

std::unique_ptr<IStrategiesInitializer> createStrategiesInitializer(const Platform device) {
  switch (device) {
  // This is the interface to control which device strategies to use.
  // Remove the Device[X]Initializer and directories:
  // include/Device[X]/, src/Device[X]/ when a device is no longer supported.
  // Benefits: It makes dialect libraries decouple from Device[X] libs.
  case Platform::Device1:
    return std::make_unique<Device1Initializer>();
  case Platform::Device2:
    return nullptr;
  }
  llvm::errs() << "Error: Unknown platform in StrategiesInitializer" << "\n";
  std::exit(1);
}

class StrategiesExtension final
    : public mlir::DialectExtension<StrategiesExtension, mlir::toy::ToyDialect> {
public:
  explicit StrategiesExtension(const Platform device) : _device(device) {
  }

  void apply(mlir::MLIRContext *context, mlir::toy::ToyDialect *) const override {
    const auto strategiesInitializer = createStrategiesInitializer(_device);
    // Initialize the strategies of one device in the context.
    // Initialize means to set cache to dialect, see Init.cpp.
    strategiesInitializer->initialize(context);
  }

private:
  Platform _device;
};

void toy::registerToyStrategies(mlir::DialectRegistry &registry, Platform device) {
  registry.addExtension(mlir::TypeID::get<StrategiesExtension>(), std::make_unique<StrategiesExtension>(device));
}
