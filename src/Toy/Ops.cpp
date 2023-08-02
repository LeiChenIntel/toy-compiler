#include "Toy/Dialect.h"

#define GET_OP_CLASSES
#include "Toy/Ops.cpp.inc"

namespace mlir {
namespace toy {
//
// FuncOp
//

void FuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   llvm::StringRef name, mlir::FunctionType type,
                   llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  // FunctionOpInterface provides a convenient `build` method that will populate
  // the state of our FuncOp, and create an entry block.
  buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

} // namespace toy
} // namespace mlir
