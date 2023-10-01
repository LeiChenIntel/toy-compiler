//===- MLIRGen.cpp - MLIR Generation from a Toy AST -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple IR generation targeting MLIR from a Module AST
// for the Toy language.
//
//===----------------------------------------------------------------------===//

#include "ToyFrontend/MLIRGen.h"
#include "Toy/Dialect.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/ScopedHashTable.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Verifier.h>

#include <numeric>

using namespace toy;
using namespace mlir::toy;

namespace {

class MLIRGenImpl {
public:
  MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}

  mlir::ModuleOp mlirGen(ModuleAST &moduleAST) {
    // We create an empty MLIR module and codegen functions one at a time and
    // add them to the module.
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

    for (FunctionAST &f : moduleAST)
      mlirGen(f);

    // Verify the module after we have finished constructing it, this will check
    // the structural properties of the IR and invoke any specific verifiers we
    // have on the Toy operations.
    if (failed(mlir::verify(theModule))) {
      theModule.emitError("module verification error");
      return nullptr;
    }

    return theModule;
  }

private:
  /// Build a tensor type from a list of shape dimensions.
  mlir::Type getType(llvm::ArrayRef<int64_t> shape) {
    // If the shape is empty, then this type is unranked.
    if (shape.empty())
      return mlir::UnrankedTensorType::get(builder.getF64Type());

    // Otherwise, we use the given shape.
    return mlir::RankedTensorType::get(shape, builder.getF64Type());
  }

  /// Build an MLIR type from a Toy AST variable type (forward to the generic
  /// getType above).
  mlir::Type getType(const VarType &type) { return getType(type.shape); }

  /// Helper conversion for a Toy AST location to an MLIR location.
  mlir::Location loc(const Location &loc) {
    return mlir::FileLineColLoc::get(builder.getStringAttr(*loc.file), loc.line,
                                     loc.col);
  }

  /// Emit a binary operation
  mlir::Value mlirGen(BinaryExprAST &binop) {
    // First emit the operations for each side of the operation before emitting
    // the operation itself. For example if the expression is `a + foo(a)`
    // 1) First it will visiting the LHS, which will return a reference to the
    //    value holding `a`. This value should have been emitted at declaration
    //    time and registered in the symbol table, so nothing would be
    //    codegen'd. If the value is not in the symbol table, an error has been
    //    emitted and nullptr is returned.
    // 2) Then the RHS is visited (recursively) and a call to `foo` is emitted
    //    and the result value is returned. If an error occurs we get a nullptr
    //    and propagate.
    //

    mlir::Value lhs = mlirGen(*binop.getLHS());
    if (!lhs)
      return nullptr;
    mlir::Value rhs = mlirGen(*binop.getRHS());
    if (!rhs)
      return nullptr;
    auto location = loc(binop.loc());

    // Derive the operation name from the binary operator. At the moment we only
    // support '+' and '*'.
    switch (binop.getOp()) {
    case '+':
      return builder.create<AddOp>(location, lhs, rhs);
    case tok_dot_multiplication:
      return builder.create<MulOp>(location, lhs, rhs);
    }

    emitError(location, "invalid binary operator '") << binop.getOp() << "'";
    return nullptr;
  }

  /// This is a reference to a variable in an expression. The variable is
  /// expected to have been declared and so should have a value in the symbol
  /// table, otherwise emit an error and return nullptr.
  mlir::Value mlirGen(VariableExprAST &expr) {
    if (auto variable = symbolTable.lookup(expr.getName()))
      return variable;

    emitError(loc(expr.loc()), "error: unknown variable '")
        << expr.getName() << "'";
    return nullptr;
  }

  /// Recursive helper function to accumulate the data that compose an array
  /// literal. It flattens the nested structure in the supplied vector. For
  /// example with this array:
  ///  [[1, 2], [3, 4]]
  /// we will generate:
  ///  [ 1, 2, 3, 4 ]
  /// Individual numbers are represented as doubles.
  /// Attributes are the way MLIR attaches constant to operations.
  void collectData(ExprAST &expr, std::vector<double> &data) {
    if (auto *lit = llvm::dyn_cast<LiteralExprAST>(&expr)) {
      for (auto &value : lit->getValues())
        collectData(*value, data);
      return;
    }

    assert(llvm::isa<NumberExprAST>(expr) && "expected literal or number expr");
    data.push_back(llvm::cast<NumberExprAST>(expr).getValue());
  }

  /// Emit a literal/constant array. It will be emitted as a flattened array of
  /// data in an Attribute attached to a `toy.constant` operation.
  /// See documentation on [Attributes](LangRef.md#attributes) for more details.
  /// Here is an excerpt:
  ///
  ///   Attributes are the mechanism for specifying constant data in MLIR in
  ///   places where a variable is never allowed [...]. They consist of a name
  ///   and a concrete attribute value. The set of expected attributes, their
  ///   structure, and their interpretation are all contextually dependent on
  ///   what they are attached to.
  ///
  /// Example, the source level statement:
  ///   var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  /// will be converted to:
  ///   %0 = "toy.constant"() {value: dense<tensor<2x3xf64>,
  ///     [[1.000000e+00, 2.000000e+00, 3.000000e+00],
  ///      [4.000000e+00, 5.000000e+00, 6.000000e+00]]>} : () -> tensor<2x3xf64>
  ///
  mlir::Value mlirGen(LiteralExprAST &lit) {
    auto type = getType(lit.getDims());

    // The attribute is a vector with a floating point value per element
    // (number) in the array, see `collectData()` below for more details.
    std::vector<double> data;
    data.reserve(std::accumulate(lit.getDims().begin(), lit.getDims().end(), 1,
                                 std::multiplies<int>()));
    collectData(lit, data);

    // The type of this attribute is tensor of 64-bit floating-point with the
    // shape of the literal.
    mlir::Type elementType = builder.getF64Type();
    auto dataType = mlir::RankedTensorType::get(lit.getDims(), elementType);

    // This is the actual attribute that holds the list of values for this
    // tensor literal.
    auto dataAttribute =
        mlir::DenseElementsAttr::get(dataType, llvm::ArrayRef(data));

    // Build the MLIR op `toy.constant`. This invokes the `ConstantOp::build`
    // method.
    return builder.create<ConstantOp>(loc(lit.loc()), type, dataAttribute);
  }

  /// Emit a constant for a single number
  /// Data is processed in frontend module to avoid additional build definition
  /// in Toy ConstantOp
  mlir::Value mlirGen(NumberExprAST &num) {
    auto dataType = mlir::RankedTensorType::get({}, builder.getF64Type());
    auto dataAttribute = mlir::DenseElementsAttr::get(dataType, num.getValue());
    return builder.create<ConstantOp>(loc(num.loc()), dataType, dataAttribute);
  }

  /// Dispatch codegen for the right expression subclass using RTTI.
  mlir::Value mlirGen(ExprAST &expr) {
    switch (expr.getKind()) {
    case toy::ExprAST::Expr_BinOp:
      return mlirGen(llvm::cast<BinaryExprAST>(expr));
    case toy::ExprAST::Expr_Var:
      return mlirGen(llvm::cast<VariableExprAST>(expr));
    case toy::ExprAST::Expr_Literal:
      return mlirGen(llvm::cast<LiteralExprAST>(expr));
    case toy::ExprAST::Expr_Call:
      return mlirGen(llvm::cast<CallExprAST>(expr));
    case toy::ExprAST::Expr_Num:
      return mlirGen(llvm::cast<NumberExprAST>(expr));
    default:
      emitError(loc(expr.loc()))
          << "MLIR codegen encountered an unhandled expr kind '"
          << llvm::Twine(expr.getKind()) << "'";
      return nullptr;
    }
  }

  /// Emit a print expression. It emits specific operations for two builtins:
  /// transpose(x) and print(x).
  mlir::LogicalResult mlirGen(PrintExprAST &call) {
    auto arg = mlirGen(*call.getArg());
    if (!arg)
      return mlir::failure();

    builder.create<PrintOp>(loc(call.loc()), arg);
    return mlir::success();
  }

  /// Emit a return operation. This will return failure if any generation fails.
  mlir::LogicalResult mlirGen(ReturnExprAST &ret) {
    auto location = loc(ret.loc());

    // 'return' takes an optional expression, handle that case here.
    mlir::Value expr = nullptr;
    if (ret.getExpr().has_value()) {
      if (!(expr = mlirGen(**ret.getExpr())))
        return mlir::failure();
    }

    // Otherwise, this return operation has zero operands.
    builder.create<ReturnOp>(location, expr ? llvm::ArrayRef(expr)
                                            : llvm::ArrayRef<mlir::Value>());
    return mlir::success();
  }

  /// Declare a variable in the current scope, return success if the variable
  /// wasn't declared yet.
  mlir::LogicalResult declare(llvm::StringRef var, mlir::Value value) {
    if (symbolTable.count(var))
      return mlir::failure();
    symbolTable.insert(var, value);
    return mlir::success();
  }

  /// Handle a variable declaration, we'll codegen the expression that forms the
  /// initializer and record the value in the symbol table before returning it.
  /// Future expressions will be able to reference this variable through symbol
  /// table lookup.
  mlir::Value mlirGen(VarDeclExprAST &vardecl) {
    auto *init = vardecl.getInitVal();
    if (!init) {
      emitError(loc(vardecl.loc()),
                "missing initializer in variable declaration");
      return nullptr;
    }

    mlir::Value value = mlirGen(*init);
    if (!value)
      return nullptr;

    // We have the initializer value, but in case the variable was declared
    // with specific shape, we emit a "reshape" operation. It will get
    // optimized out later as needed.
    // We can handle constant as a special case, but writing code in the way
    // makes the pipeline more general.
    if (!vardecl.getType().shape.empty()) {
      value = builder.create<ReshapeOp>(loc(vardecl.loc()),
                                        getType(vardecl.getType()), value);
    }

    // Register the value in the symbol table.
    if (failed(declare(vardecl.getName(), value)))
      return nullptr;
    return value;
  }

  /// Create the prototype for an MLIR function with as many arguments as the
  /// provided Toy AST prototype.
  FuncOp mlirGen(PrototypeAST &proto) {
    auto location = loc(proto.loc());

    // This is a generic function, the return type will be inferred later.
    // Arguments type are uniformly unranked tensors.
    llvm::SmallVector<mlir::Type, 4> argTypes(proto.getArgs().size(),
                                              getType(VarType{}));
    auto funcType = builder.getFunctionType(argTypes, std::nullopt);
    return builder.create<FuncOp>(location, proto.getName(), funcType);
  }

  /// Codegen a list of expression, return failure if one of them hit an error.
  mlir::LogicalResult mlirGen(ExprASTList &blockAST) {
    llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(
        symbolTable);
    for (auto &expr : blockAST) {
      // Specific handling for variable declarations, return statement, and
      // print. These can only appear in block list and not in nested
      // expressions.
      if (auto *vardecl = llvm::dyn_cast<VarDeclExprAST>(expr.get())) {
        if (!mlirGen(*vardecl))
          return mlir::failure();
        continue;
      }
      if (auto *ret = llvm::dyn_cast<ReturnExprAST>(expr.get()))
        return mlirGen(*ret);
      if (auto *print = llvm::dyn_cast<PrintExprAST>(expr.get())) {
        if (mlir::failed(mlirGen(*print)))
          return mlir::success();
        continue;
      }

      // Generic expression dispatch codegen.
      if (!mlirGen(*expr))
        return mlir::failure();
    }
    return mlir::success();
  }

  /// Emit a new function and add it to the MLIR module.
  FuncOp mlirGen(FunctionAST &funcAST) {
    // Create a scope in the symbol table to hold variable declarations.
    llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(
        symbolTable);

    // Create an MLIR function for the given prototype.
    builder.setInsertionPointToEnd(theModule.getBody());
    FuncOp function = mlirGen(*funcAST.getProto());
    if (!function)
      return nullptr;

    // Let's start the body of the function now!
    mlir::Block &entryBlock = function.front();
    auto protoArgs = funcAST.getProto()->getArgs();

    // Declare all the function arguments in the symbol table.
    for (const auto nameValue :
         llvm::zip(protoArgs, entryBlock.getArguments())) {
      if (failed(declare(std::get<0>(nameValue)->getName(),
                         std::get<1>(nameValue))))
        return nullptr;
    }

    // Set the insertion point in the builder to the beginning of the function
    // body, it will be used throughout the codegen to create operations in this
    // function.
    builder.setInsertionPointToStart(&entryBlock);

    // Emit the body of the function.
    if (mlir::failed(mlirGen(*funcAST.getBody()))) {
      function.erase();
      return nullptr;
    }

    // Implicitly return void if no return statement was emitted.
    // FIXME: we may fix the parser instead to always return the last expression
    // (this would possibly help the REPL case later)
    ReturnOp returnOp;
    if (!entryBlock.empty())
      returnOp = llvm::dyn_cast<ReturnOp>(entryBlock.back());
    if (!returnOp) {
      builder.create<ReturnOp>(loc(funcAST.getProto()->loc()), std::nullopt);
    } else if (returnOp.getNumOperands() != 0) {
      // Otherwise, if this return operation has an operand then add a result to
      // the function.
      function.setType(builder.getFunctionType(
          function.getFunctionType().getInputs(), getType(VarType{})));
    }

    return function;
  }

private:
  mlir::ModuleOp theModule;
  mlir::OpBuilder builder;
  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;
};

} // namespace

namespace toy {

mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                          ModuleAST &moduleAST) {
  return MLIRGenImpl(context).mlirGen(moduleAST);
}

} // namespace toy
