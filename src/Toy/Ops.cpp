#include "Toy/Dialect.h"

#include <mlir/IR/FunctionImplementation.h>

#define GET_OP_CLASSES
#include "Toy/Ops.cpp.inc"

namespace mlir {
namespace toy {

//
// AddOp
//

mlir::LogicalResult AddOp::inferReturnTypes(
    ::mlir::MLIRContext *ctx, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attrs,
    ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  const auto loc = location.value_or(mlir::UnknownLoc::get(ctx));
  AddOpAdaptor add(operands, attrs);
  if (mlir::failed(add.verify(loc))) {
    return mlir::failure();
  }

  const auto inLhsType = add.getLhs().getType().cast<mlir::ShapedType>();
  const auto inRhsType = add.getRhs().getType().cast<mlir::ShapedType>();

  mlir::Type outType;
  if (!inLhsType.hasRank() || !inRhsType.hasRank()) {
    outType = mlir::UnrankedTensorType::get(inLhsType.getElementType());
  } else {
    outType = mlir::RankedTensorType::get(inLhsType.getShape(),
                                          inLhsType.getElementType());
  }
  inferredReturnTypes.push_back(outType);
  return mlir::success();
}

//
// MulOp
//

mlir::LogicalResult MulOp::inferReturnTypes(
    mlir::MLIRContext *ctx, std::optional<::mlir::Location> location,
    mlir::ValueRange operands, mlir::DictionaryAttr attrs,
    mlir::RegionRange regions,
    llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  const auto loc = location.value_or(mlir::UnknownLoc::get(ctx));
  MulOpAdaptor mul(operands, attrs);
  if (mlir::failed(mul.verify(loc))) {
    return mlir::failure();
  }

  const auto inLhsType = mul.getLhs().getType().cast<mlir::ShapedType>();
  const auto inRhsType = mul.getRhs().getType().cast<mlir::ShapedType>();

  mlir::Type outType;
  if (!inLhsType.hasRank() || !inRhsType.hasRank()) {
    outType = mlir::UnrankedTensorType::get(inLhsType.getElementType());
  } else {
    outType = mlir::RankedTensorType::get(inLhsType.getShape(),
                                          inLhsType.getElementType());
  }
  inferredReturnTypes.push_back(outType);
  return mlir::success();
}

//
// FuncOp
//

void FuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   llvm::StringRef name, mlir::FunctionType type,
                   llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  // FunctionOpInterface provides a convenient `build` method that will populate
  // the state of our FuncOp, and create an entry block.
  buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

mlir::ParseResult FuncOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &result) {
  auto buildFuncType =
      [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
         llvm::ArrayRef<mlir::Type> results,
         mlir::function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return mlir::function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(mlir::OpAsmPrinter &p) {
  mlir::function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

} // namespace toy
} // namespace mlir
