#include "Toy/Dialect.h"

#include <mlir/IR/FunctionImplementation.h>

#define GET_OP_CLASSES
#include "Toy/Ops.cpp.inc"

namespace mlir {
namespace toy {

//
// AddOp
//

mlir::OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  // Only an existing value or attribute can be returned.
  // Cannot return created operation. Therefore, lhs operation is reset.
  const auto lhs = getLhs();
  const auto rhs = getRhs();
  if (!lhs.getDefiningOp<toy::ConstantOp>() ||
      !rhs.getDefiningOp<toy::ConstantOp>()) {
    return nullptr;
  }

  ConstantOp lhsOp = lhs.getDefiningOp<toy::ConstantOp>();
  const auto lhsCstAttr = lhsOp.getValueAttr();
  const int len = lhsOp.getType().dyn_cast<TensorType>().getNumElements();
  const auto lhsType = lhsOp.getType();
  const auto lhsVal = lhsCstAttr.getValues<double>();

  ConstantOp rhsOp = rhs.getDefiningOp<toy::ConstantOp>();
  const auto rhsCstAttr = rhsOp.getValueAttr();
  const auto rhsVal = rhsCstAttr.getValues<double>();

  std::vector<double> resVal;
  for (int i = 0; i < len; i++) {
    double v = lhsVal[i] + rhsVal[i];
    resVal.push_back(v);
  }

  const auto resCstAttr =
      mlir::DenseElementsAttr::get(lhsType, llvm::ArrayRef(resVal));
  lhsOp.setValueAttr(resCstAttr);
  return getLhs();
}

mlir::LogicalResult AddOp::verify() {
  const auto lhsType = getLhs().getType().dyn_cast<mlir::ShapedType>();
  const auto rhsType = getRhs().getType().dyn_cast<mlir::ShapedType>();

  if (lhsType.getElementType() != rhsType.getElementType()) {
    mlir::emitError(getLoc(), "AddOp: Input element type mismatch");
    return mlir::failure();
  }

  if (!lhsType.hasRank() || !rhsType.hasRank()) {
    return mlir::success();
  }
  if (lhsType.getNumElements() != rhsType.getNumElements()) {
    mlir::emitError(getLoc(), "AddOp: Input element numbers mismatch");
    return mlir::failure();
  }

  return mlir::success();
}

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
// MatmulOp
//

mlir::OpFoldResult MatmulOp::fold(FoldAdaptor adaptor) {
  const auto lhs = getLhs();
  const auto rhs = getRhs();
  if (!lhs.getDefiningOp<toy::ConstantOp>() ||
      !rhs.getDefiningOp<toy::ConstantOp>()) {
    return nullptr;
  }

  ConstantOp lhsOp = lhs.getDefiningOp<toy::ConstantOp>();
  const auto lhsCstAttr = lhsOp.getValueAttr();
  const int len = lhsOp.getType().dyn_cast<TensorType>().getNumElements();
  const auto lhsType = lhsOp.getType();
  const auto lhsVal = lhsCstAttr.getValues<double>();

  ConstantOp rhsOp = rhs.getDefiningOp<toy::ConstantOp>();
  const auto rhsCstAttr = rhsOp.getValueAttr();
  const auto rhsVal = rhsCstAttr.getValues<double>();

  std::vector<double> resVal;
  for (int i = 0; i < len; i++) {
    double v = lhsVal[i] * rhsVal[i];
    resVal.push_back(v);
  }

  const auto resCstAttr =
      mlir::DenseElementsAttr::get(lhsType, llvm::ArrayRef(resVal));
  lhsOp.setValueAttr(resCstAttr);
  return getLhs();
}

mlir::LogicalResult MatmulOp::verify() {
  const auto lhsType = getLhs().getType().dyn_cast<mlir::ShapedType>();
  const auto rhsType = getRhs().getType().dyn_cast<mlir::ShapedType>();
  const auto lhsShape = lhsType.getShape();
  const auto rhsShape = rhsType.getShape();

  if (lhsType.getElementType() != rhsType.getElementType()) {
    mlir::emitError(getLoc(), "MatmulOp: Input element type mismatch");
    return mlir::failure();
  }

  if (!lhsType.hasRank() || !rhsType.hasRank()) {
    return mlir::success();
  }
  if (lhsType.getNumElements() != rhsType.getNumElements()) {
    mlir::emitError(getLoc(), "MatmulOp: Input element numbers mismatch");
    return mlir::failure();
  }
  // 矩阵乘法基本约束
  if (lhsShape[1] != rhsShape[0]){
    mlir::emitError(getLoc(),"MatmulOp: Input matrix shape mismatch");
  }
  return mlir::success();
}

mlir::LogicalResult MatmulOp::inferReturnTypes(
    mlir::MLIRContext *ctx, std::optional<::mlir::Location> location,
    mlir::ValueRange operands, mlir::DictionaryAttr attrs,
    mlir::RegionRange regions,
    llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes){

    MatmulOpAdaptor mul(operands, attrs);
    const auto inLhsType = mul.getLhs().getType().cast<mlir::ShapedType>();
    const auto inRhsType = mul.getRhs().getType().cast<mlir::ShapedType>();

    mlir::Type outType;
    // hasRank貌似是判断是否指定了shape的作用
    if (!inLhsType.hasRank() || !inRhsType.hasRank()){
      outType = mlir::UnrankedTensorType::get(inLhsType.getElementType());
    }
    else{
      const auto inLhsShape =  inLhsType.getShape();
      const auto inRhsShape = inRhsType.getShape();
      const auto resultShape = {inLhsShape[0],inRhsShape[1]};
      outType = mlir::RankedTensorType::get(resultShape,
                                      inLhsType.getElementType());
    }
    inferredReturnTypes.push_back(outType);
    return mlir::success();
}

//
// MulOp
//

mlir::OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  const auto lhs = getLhs();
  const auto rhs = getRhs();
  if (!lhs.getDefiningOp<toy::ConstantOp>() ||
      !rhs.getDefiningOp<toy::ConstantOp>()) {
    return nullptr;
  }

  ConstantOp lhsOp = lhs.getDefiningOp<toy::ConstantOp>();
  const auto lhsCstAttr = lhsOp.getValueAttr();
  const int len = lhsOp.getType().dyn_cast<TensorType>().getNumElements();
  const auto lhsType = lhsOp.getType();
  const auto lhsVal = lhsCstAttr.getValues<double>();

  ConstantOp rhsOp = rhs.getDefiningOp<toy::ConstantOp>();
  const auto rhsCstAttr = rhsOp.getValueAttr();
  const auto rhsVal = rhsCstAttr.getValues<double>();

  std::vector<double> resVal;
  for (int i = 0; i < len; i++) {
    double v = lhsVal[i] * rhsVal[i];
    resVal.push_back(v);
  }

  const auto resCstAttr =
      mlir::DenseElementsAttr::get(lhsType, llvm::ArrayRef(resVal));
  lhsOp.setValueAttr(resCstAttr);
  return getLhs();
}

mlir::LogicalResult MulOp::verify() {
  const auto lhsType = getLhs().getType().dyn_cast<mlir::ShapedType>();
  const auto rhsType = getRhs().getType().dyn_cast<mlir::ShapedType>();

  if (lhsType.getElementType() != rhsType.getElementType()) {
    mlir::emitError(getLoc(), "MulOp: Input element type mismatch");
    return mlir::failure();
  }

  if (!lhsType.hasRank() || !rhsType.hasRank()) {
    return mlir::success();
  }
  if (lhsType.getNumElements() != rhsType.getNumElements()) {
    mlir::emitError(getLoc(), "MulOp: Input element numbers mismatch");
    return mlir::failure();
  }

  return mlir::success();
}

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
// ReshapeOp
//

mlir::LogicalResult ReshapeOp::verify() {
  const auto inType = getInput().getType();
  if (!inType.hasRank()) {
    mlir::emitError(getLoc(), "ReshapeOp: Not support dynamic shape");
    return mlir::failure();
  }

  const auto outType = getOutput().getType();
  if (inType.getElementType() != outType.getElementType()) {
    mlir::emitError(getLoc(), "ReshapeOp: Input, output type mismatch");
    return mlir::failure();
  }
  if (inType.getNumElements() != outType.getNumElements() &&
      inType.getNumElements() != 1) {
    // Only support constant broadcast from 1-dim to multi-dim
    mlir::emitError(getLoc(),
                    "ReshapeOp: Input, output element numbers mismatch");
    return mlir::failure();
  }

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
