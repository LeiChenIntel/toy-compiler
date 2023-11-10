#include "Conversion/Passes.h"
#include "Toy/Dialect.h"
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
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/TargetSelect.h>

struct MemRef {
  double *allocated = nullptr;
  double *aligned = nullptr;
  intptr_t offset = 0;
  intptr_t size;
  intptr_t stride;
};

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

  mlir::OpPassManager &optPM = pm.nest<mlir::toy::FuncOp>();
  optPM.addPass(mlir::createCanonicalizerPass());
  optPM.addPass(mlir::createCSEPass());
  pm.addPass(mlir::toy::createConvertToyToMidPass());
  mlir::OpPassManager &optPM2 = pm.nest<mlir::func::FuncOp>();
  optPM2.addPass(mlir::createCanonicalizerPass());
  optPM2.addPass(mlir::createCSEPass());
  optPM2.addPass(mlir::createLoopFusionPass());
  optPM2.addPass(mlir::createAffineScalarReplacementPass());
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
  llvm::errs() << *llvmModule << "\n";

  // Create and invoke a MLIR execution engine.
  mlir::ExecutionEngineOptions engineOptions;
  engineOptions.transformer = optPipeline;
  auto maybeEngine = mlir::ExecutionEngine::create(*module, engineOptions);
  if (!maybeEngine) {
    llvm::errs() << "Fail to construct an execution engine\n";
  }

  std::unique_ptr<mlir::ExecutionEngine> jit = std::move(maybeEngine.get());
  std::vector<double> inputData1 = {0, 1, 2};
  MemRef input1;
  input1.allocated = inputData1.data();
  input1.aligned = inputData1.data();
  input1.size = 3;
  input1.stride = 0;

  std::vector<double> inputData2 = {3, 4, 5};
  MemRef input2;
  input2.allocated = inputData2.data();
  input2.aligned = inputData2.data();
  input2.size = 3;
  input2.stride = 0;

  MemRef input3;
  double *p = (double *)malloc(3 * 8);
  input3.allocated = p;
  input3.aligned = p;
  input3.size = 3;
  input3.stride = 0;

  // Function name is changed without emit_c_interface
  // Compiler is trying to add ciface_ before function name why not
  // _mlir_ciface_? Need to invoke ciface+name and use invokePacked to avoid
  // issues
  llvm::SmallVector<void *> argsArray;
  argsArray.push_back(&input1);
  argsArray.push_back(&input2);
  argsArray.push_back(&input3);

  llvm::Error error = jit->invokePacked("print_tensor", argsArray);
  if (error) {
    llvm::errs() << "Fail to run JIT\n";
  }

  double *ptr = p;
  for (int i = 0; i < 3; i++) {
    printf("%f ", *ptr);
    ptr++;
  }

  return 0;
}
