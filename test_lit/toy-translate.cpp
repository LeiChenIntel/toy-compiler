#include "Conversion/Passes.h"
#include "Toy/Dialect.h"
#include "ToyFrontend/MLIRGen.h"
#include "ToyFrontend/Parser.h"

#include <mlir/Dialect/Affine/Passes.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>

namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));
// Canonicalization, CSE are included in this option
static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));
namespace {
enum Action {
  None,
  DumpAST,
  DumpMLIR,
  DumpMLIRMid,
  DumpMLIRLLVM,
  DumpLLVMIR,
  RunJIT
};
} // namespace

static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
    cl::values(clEnumValN(DumpMLIRMid, "mlir-mid",
                          "output the MLIR dump after lowering")),
    cl::values(clEnumValN(DumpMLIRLLVM, "mlir-llvm",
                          "output the MLIR dump after llvm lowering")),
    cl::values(clEnumValN(DumpLLVMIR, "llvm", "output the LLVM IR dump")),
    cl::values(
        clEnumValN(RunJIT, "jit",
                   "JIT the code and run it by invoking the main function")));

/// Returns a Toy AST resulting from parsing the file or a nullptr on error.
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

int dumpMLIR(mlir::MLIRContext &ctx,
             mlir::OwningOpRef<mlir::ModuleOp> &module) {
  // Convert to AST IR
  auto moduleAST = parseInputFile(inputFilename);
  if (!moduleAST) {
    return 1;
  }
  // Convert to MLIR format with no optimization
  module = mlirGen(ctx, *moduleAST);
  if (!module) {
    return 1;
  }
  // Add optimization and lowering pass
  mlir::PassManager pm(&ctx);
  // Apply any generic pass manager command line options and run the pipeline.
  applyPassManagerCLOptions(pm);

  // Check to see what granularity of MLIR we are compiling to.
  bool isLoweringToMid = emitAction >= Action::DumpMLIRMid;
  bool isLoweringToLLVM = emitAction >= Action::DumpMLIRLLVM;

  if (enableOpt || isLoweringToMid) {
    mlir::OpPassManager &optPM = pm.nest<mlir::toy::FuncOp>();
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
  }

  if (isLoweringToMid) {
    pm.addPass(mlir::toy::createConvertToyToMidPass());

    if (enableOpt) {
      // Add a few cleanups post lowering.
      mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
      optPM.addPass(mlir::createCanonicalizerPass());
      optPM.addPass(mlir::createCSEPass());

      // Add Affine optimizations.
      optPM.addPass(mlir::createLoopFusionPass());
      optPM.addPass(mlir::createAffineScalarReplacementPass());
    }
  }

  if (isLoweringToLLVM) {
    pm.addPass(mlir::toy::createConvertMidToLLVMPass());
  }

  if (mlir::failed(pm.run(*module)))
    return 4;
  return 0;
}

int dumpLLVMIR(mlir::ModuleOp module) {
  // Register the translation to LLVM IR with the MLIR context.
  mlir::registerLLVMDialectTranslation(*(module->getContext()));

  // Convert the module to LLVM IR in a new LLVM IR context.
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

  /// Optionally run an optimization pipeline over the llvm module.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }
  llvm::errs() << *llvmModule << "\n";
  return 0;
}

int runJIT(mlir::ModuleOp module) {
  // Register the translation from MLIR to LLVM IR, which must happen before we
  // can JIT-compile.
  mlir::registerLLVMDialectTranslation(*(module->getContext()));

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // An optimization pipeline to use within the execution engine.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  mlir::ExecutionEngineOptions engineOptions;
  engineOptions.transformer = optPipeline;
  auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();

  // Invoke the JIT-compiled function.
  auto invocationResult = engine->invokePacked("main");
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }

  return 0;
}

int dumpAST() {
  auto moduleAST = parseInputFile(inputFilename);
  if (!moduleAST) {
    return 1;
  }
  dump(*moduleAST);
  return 0;
}

int main(int argc, char **argv) {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  // Can use this option to print IR
  mlir::registerPassManagerCLOptions();
  cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");

  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::toy::ToyDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module;

  switch (emitAction) {
  case Action::DumpAST:
    return dumpAST();
  case Action::DumpMLIR:
  case Action::DumpMLIRMid:
  case Action::DumpMLIRLLVM:
  case Action::DumpLLVMIR:
  case Action::RunJIT:
    // Need to translate IR to LLVM for LLVMIR and JIT
    dumpMLIR(context, module);
    break;
  default:
    llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  }

  if (emitAction <= Action::DumpMLIRLLVM) {
    module->dump();
    return 0;
  }

  // Check to see if we are compiling to LLVM IR.
  if (emitAction == Action::DumpLLVMIR) {
    return dumpLLVMIR(*module);
  }
  // Otherwise, we must be running the jit.
  if (emitAction == Action::RunJIT) {
    return runJIT(*module);
  }

  llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  return -1;
}
