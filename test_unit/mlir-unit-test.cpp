#include <gtest/gtest.h>
#include <iostream>
#include <mlir/IR/BuiltinTypes.h>

TEST(MLIR, BuiltinTypes) {
  mlir::MLIRContext *ctx = new mlir::MLIRContext;
  mlir::FloatType f16Type = mlir::Float16Type::get(ctx);
  f16Type.dump();
  EXPECT_EQ(f16Type.getWidth(), 16);
}
