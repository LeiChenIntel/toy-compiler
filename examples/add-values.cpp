#include <mlir-c/Conversion.h>
#include <mlir-c/ExecutionEngine.h>
#include <mlir-c/IR.h>
#include <mlir-c/RegisterEverything.h>

#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>

#include <stdio.h>

void testAddValuesCAPI() {
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
  mlirPassManagerAddOwnedPass(pm, mlirCreateConversionConvertFuncToLLVMPass());
  mlirOpPassManagerAddOwnedPass(
      opm, mlirCreateConversionArithToLLVMConversionPass());
  MlirLogicalResult status =
      mlirPassManagerRunOnOp(pm, mlirModuleGetOperation(module));
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
  int input = 41;
  int result = -1;
  void *args[2] = {&input, &result};
  if (mlirLogicalResultIsFailure(mlirExecutionEngineInvokePacked(
          jit, mlirStringRefCreateFromCString("add"), args))) {
    fprintf(stderr, "Execution engine creation failed");
    abort();
  }

  // CHECK: Input: 41 Result: 82
  printf("TEST: Add Value C API\n");
  printf("Input: %d Result: %d\n", input, result);
  mlirExecutionEngineDestroy(jit);
  mlirModuleDestroy(module);
  mlirContextDestroy(ctx);
}

void testAddValuesCPP() {
  // Register dialects, translations
  mlir::DialectRegistry registry;
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::BuiltinDialect,
                  mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                  mlir::LLVM::LLVMDialect>();
  mlir::MLIRContext ctx(registry);

  std::string moduleStr = R"mlir(
  func.func @foo(%arg0 : i32) -> i32 attributes { llvm.emit_c_interface } {
    %res = arith.addi %arg0, %arg0 : i32
    return %res : i32
  }
  )mlir";
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(moduleStr, &ctx);
  if (!module) {
    llvm::errs() << "Fail to parse string to module\n";
    return;
  }

  // Register and run passes
  // Lower to LLVM IR
  mlir::PassManager pm(module->getContext());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  mlir::LogicalResult res = pm.run(*module);
  if (res.failed()) {
    llvm::errs() << "Fail to run lowering pass to LLVM\n";
  }

  // Lower to LLVM and register JIT
  auto maybeEngine = mlir::ExecutionEngine::create(*module);
  if (!maybeEngine) {
    llvm::errs() << "Fail to construct an execution engine\n";
  }
  std::unique_ptr<mlir::ExecutionEngine> jit = std::move(maybeEngine.get());
  int input = 42;
  int result = 0;
  llvm::Error error =
      jit->invoke("foo", input, mlir::ExecutionEngine::Result<int>(result));
  if (error) {
    llvm::errs() << "Fail to run JIT\n";
  }

  // CHECK: Input: 42 Result: 84
  printf("TEST: Add Value CPP API\n");
  printf("Input: %d Result: %d\n", input, result);
}

int main() {
  testAddValuesCAPI();
  testAddValuesCPP();
  return 0;
}
