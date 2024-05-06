#ifndef TOY_COMPILER_DIALECT_H
#define TOY_COMPILER_DIALECT_H

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>

#include "Toy/Dialect.h.inc"
#include "Toy/ToyInterface.h.inc"

#define GET_OP_CLASSES
#include "Toy/Ops.h.inc"

#endif // TOY_COMPILER_DIALECT_H
