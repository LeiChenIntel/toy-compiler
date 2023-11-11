#include "Conversion/Passes.h"
#include "Toy/Dialect.h"
#include "ToyFrontend/Container.h"
#include "ToyFrontend/MLIRGen.h"
#include "ToyFrontend/Parser.h"

#include <mlir/Dialect/Affine/Passes.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/AsmState.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/TargetSelect.h>

namespace cl = llvm::cl;
static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

std::unique_ptr<toy::ModuleAST> parseInputFile(llvm::StringRef filename) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return nullptr;
  }
  auto buffer = fileOrErr.get()->getBuffer();
  toy::LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
  toy::Parser parser(lexer);
  return parser.parseModule();
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");

  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<mlir::toy::ToyDialect>();

  auto moduleAST = parseInputFile(inputFilename);
  if (!moduleAST) {
    llvm::errs() << "Fail to lower to AST\n";
    return -1;
  }
  auto module = mlirGen(ctx, *moduleAST);
  if (!module) {
    llvm::errs() << "Fail to lower to MLIR\n";
    return -1;
  }

  mlir::PassManager pm(&ctx);
  applyPassManagerCLOptions(pm);

  mlir::OpPassManager &optPmToy = pm.nest<mlir::toy::FuncOp>();
  optPmToy.addPass(mlir::createCanonicalizerPass());
  optPmToy.addPass(mlir::createCSEPass());
  pm.addPass(mlir::toy::createConvertToyToMidPass());
  mlir::OpPassManager &optPmFunc = pm.nest<mlir::func::FuncOp>();
  optPmFunc.addPass(mlir::createCanonicalizerPass());
  optPmFunc.addPass(mlir::createCSEPass());
  optPmFunc.addPass(mlir::createLoopFusionPass());
  optPmFunc.addPass(mlir::createAffineScalarReplacementPass());
  pm.addPass(mlir::toy::createConvertMidToLLVMPass());
  if (mlir::failed(pm.run(*module))) {
    llvm::errs() << "Fail to run lowering pass to LLVM\n";
    return -1;
  }

  mlir::registerLLVMDialectTranslation(*(module->getContext()));
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/3, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }
  // Dump LLVM. Commented as default.
  // llvm::errs() << *llvmModule << "\n";

  // Create and invoke a MLIR execution engine.
  mlir::ExecutionEngineOptions engineOptions;
  engineOptions.transformer = optPipeline;
  auto maybeEngine = mlir::ExecutionEngine::create(*module, engineOptions);
  if (!maybeEngine) {
    llvm::errs() << "Fail to construct an execution engine\n";
    return -1;
  }

  std::unique_ptr<mlir::ExecutionEngine> jit = std::move(maybeEngine.get());

  std::vector<double> inputData1 = {0, 1, 2};
  MemRef input1{inputData1.data(), inputData1.data(), 3, 0};
  std::vector<double> inputData2 = {3, 4, 5};
  MemRef input2{inputData2.data(), inputData2.data(), 3, 0};
  double *p = (double *)malloc(3 * sizeof(double));
  MemRef output{p, p, 3, 0};

  llvm::SmallVector<void *> argsArray;
  argsArray.push_back(&input1);
  argsArray.push_back(&input2);
  argsArray.push_back(&output);

  llvm::Error error = jit->invokePacked("add_tensors", argsArray);
  if (error) {
    llvm::errs() << "Fail to run JIT\n";
    return -1;
  }

  // Dump output data. Values should be 3, 5, 7.
  for (int i = 0; i < 3; i++) {
    printf("%f ", p[i]);
  }

  return 0;
}
