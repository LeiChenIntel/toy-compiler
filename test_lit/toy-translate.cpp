#include "Conversion/Passes.h"
#include "Toy/Dialect.h"
#include "ToyFrontend/MLIRGen.h"
#include "ToyFrontend/Parser.h"

#include <mlir/Dialect/Affine/Passes.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>

namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));
// Canonicalization, CSE are included in this option
static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));
namespace {
enum Action { None, DumpAST, DumpMLIR, DumpMLIRMid, DumpMLIRLLVM };
} // namespace

static cl::opt<enum Action>
    emitAction("emit", cl::desc("Select the kind of output desired"),
               cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
               cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
               cl::values(clEnumValN(DumpMLIRMid, "mlir-mid",
                                     "output the MLIR dump after lowering")));

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

int dumpMLIR() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::toy::ToyDialect>();

  // Convert to AST IR
  auto moduleAST = parseInputFile(inputFilename);
  if (!moduleAST) {
    return 1;
  }
  // Convert to MLIR format with no optimization
  mlir::OwningOpRef<mlir::ModuleOp> module = mlirGen(context, *moduleAST);
  if (!module) {
    return 1;
  }
  // Add optimization and lowering pass
  mlir::PassManager pm(&context);
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

  module->dump();
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

  switch (emitAction) {
  case Action::DumpAST:
    return dumpAST();
  case Action::DumpMLIR:
  case Action::DumpMLIRMid:
    return dumpMLIR();
  default:
    llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  }

  return 0;
}
