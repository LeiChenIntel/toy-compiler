#include "Toy/Dialect.h"

#include <mlir/InitAllPasses.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::toy::ToyDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Toy optimizer tool\n", registry));
}
