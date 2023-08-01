#include "Toy/Dialect.h"
#include "Toy/Dialect.cpp.inc"

using namespace mlir;
using namespace mlir::toy;

void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Toy/Ops.cpp.inc"
      >();
}
