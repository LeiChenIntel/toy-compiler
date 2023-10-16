#include <mlir-c/Conversion.h>
#include <mlir-c/ExecutionEngine.h>
#include <mlir-c/IR.h>
#include <mlir-c/RegisterEverything.h>

#include <stdio.h>
#include <stdlib.h>

void testAddTensor() {
  MlirContext ctx = mlirContextCreate();
  MlirDialectRegistry registry = mlirDialectRegistryCreate();

  // Register dialects
  mlirRegisterAllDialects(registry);
  mlirContextAppendDialectRegistry(ctx, registry);
  mlirDialectRegistryDestroy(registry);

  MlirModule module = mlirModuleCreateParse(
      ctx, mlirStringRefCreateFromCString(
               // clang-format off
"module {                                                                    \n"
"  func.func @add(%arg0 : i32) -> i32 attributes { llvm.emit_c_interface } {     \n"
"    %res = arith.addi %arg0, %arg0 : i32                                        \n"
"    return %res : i32                                                           \n"
"  }                                                                             \n"
"}"));
  // clang-format on

  // Register and run passes
  // Lower to LLVM IR
  MlirPassManager pm = mlirPassManagerCreate(ctx);
  MlirOpPassManager opm = mlirPassManagerGetNestedUnder(
      pm, mlirStringRefCreateFromCString("func.func"));
  mlirPassManagerAddOwnedPass(pm, mlirCreateConversionConvertFuncToLLVM());
  mlirOpPassManagerAddOwnedPass(
      opm, mlirCreateConversionArithToLLVMConversionPass());
  MlirLogicalResult status = mlirPassManagerRun(pm, module);
  if (mlirLogicalResultIsFailure(status)) {
    fprintf(stderr, "Unexpected failure running pass pipeline\n");
    exit(2);
  }
  mlirPassManagerDestroy(pm);

  // Lower to LLVM and register JIT
  mlirRegisterAllLLVMTranslations(ctx);
  MlirExecutionEngine jit = mlirExecutionEngineCreate(
      module, /*optLevel=*/2, /*numPaths=*/0, /*sharedLibPaths=*/NULL,
      /*enableObjectDump=*/false);
  if (mlirExecutionEngineIsNull(jit)) {
    fprintf(stderr, "Execution engine creation failed");
    exit(2);
  }

  // Run JIT
  int input = 42;
  int result = -1;
  void *args[2] = {&input, &result};
  if (mlirLogicalResultIsFailure(mlirExecutionEngineInvokePacked(
          jit, mlirStringRefCreateFromCString("add"), args))) {
    fprintf(stderr, "Execution engine creation failed");
    abort();
  }

  // CHECK: Input: 42 Result: 84
  printf("Input: %d Result: %d\n", input, result);
  mlirExecutionEngineDestroy(jit);
  mlirModuleDestroy(module);
  mlirContextDestroy(ctx);
}

int main() {
  testAddTensor();
  return 0;
}
