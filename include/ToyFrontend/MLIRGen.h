//===----------------------------------------------------------------------===//
//
// This file declares a simple interface to perform IR generation targeting MLIR
// from a Module AST for the Toy language.
//
//===----------------------------------------------------------------------===//

#ifndef TOY_COMPILER_MLIRGEN_H
#define TOY_COMPILER_MLIRGEN_H

#include "ToyFrontend/AST.h"
#include <mlir/IR/BuiltinOps.h>

namespace toy {

mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                          ModuleAST &moduleAST);

} // namespace toy

#endif // TOY_COMPILER_MLIRGEN_H
