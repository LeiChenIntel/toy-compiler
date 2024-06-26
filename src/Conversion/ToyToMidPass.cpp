#include "Conversion/Passes.h"
#include "Toy/Dialect.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "Conversion/Passes.h.inc"

namespace {

//
// Common functions
//

/// Convert the given TensorType into the corresponding MemRefType.
static MemRefType convertTensorToMemRef(TensorType type) {
  assert(type.hasRank() && "expected only ranked shapes");
  return MemRefType::get(type.getShape(), type.getElementType());
}

/// Insert an allocation and deallocation for the given MemRefType.
static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter) {
  auto alloc = rewriter.create<memref::AllocOp>(loc, type);

  // Make sure to allocate at the beginning of the block.
  auto *parentBlock = alloc->getBlock();
  alloc->moveBefore(&parentBlock->front());

  // Make sure to deallocate this alloc at the end of the block. This is fine
  // as toy functions have no control flow.
  auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
  dealloc->moveBefore(&parentBlock->back());
  return alloc;
}

static Value createStoreOpMemRef(Operation *op, PatternRewriter &rewriter) {
  const auto tensorType = (*op->result_type_begin()).cast<TensorType>();
  const auto loc = op->getLoc();
  const auto opResVal = op->getResult(0);
  mlir::Value memRef;
  // Need to check if the result is stored to input pointers.
  bool isMemAllocated = false;

  // Find if the store operation exists.
  auto &blk = op->getParentRegion()->front();
  const auto storeOps = blk.getOps<toy::StoreOp>();
  for (auto op : storeOps) {
    if (opResVal == op.getValToStore()) {
      memRef = op.getMemref();
      isMemAllocated = true;
      rewriter.eraseOp(op);
      break;
    }
  }

  if (!isMemAllocated) {
    // Insert an allocation and deallocation for the result of this operation.
    const auto memRefType = convertTensorToMemRef(tensorType);
    memRef = insertAllocAndDealloc(memRefType, loc, rewriter);
  }

  return memRef;
}

/// This defines the function type used to process an iteration of a lowered
/// loop. It takes as input an OpBuilder, an range of memRefOperands
/// corresponding to the operands of the input operation, and the range of loop
/// induction variables for the iteration. It returns a value to store at the
/// current index of the iteration.
using LoopIterationFn = function_ref<Value(
    OpBuilder &rewriter, ValueRange memRefOperands, ValueRange loopIvs)>;

// Divide the operation operands into elements.
//
static void lowerOpToLoops(Operation *op, ValueRange operands,
                           PatternRewriter &rewriter,
                           LoopIterationFn processIteration) {
  auto tensorType = (*op->result_type_begin()).cast<TensorType>();
  auto loc = op->getLoc();

  if (op->getResults().size() != 1) {
    emitError(loc, "Only support operation with 1 result");
  }

  mlir::Value memRef = createStoreOpMemRef(op, rewriter);

  // Create a nest of affine loops, with one loop per dimension of the shape.
  // The buildAffineLoopNest function takes a callback that is used to construct
  // the body of the innermost loop given a builder, a location and a range of
  // loop induction variables.
  // lowerBounds: all zeros, steps: all ones. Then iteration is applied on each
  // element of the tensor.
  // Load and store one by one.
  SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value=*/0);
  SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);
  affine::buildAffineLoopNest(
      rewriter, loc, lowerBounds, tensorType.getShape(), steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        // Call the processing function with the rewriter, the memref operands,
        // and the loop induction variables. This function will return the value
        // to store at the current index.
        Value valueToStore = processIteration(nestedBuilder, operands, ivs);
        nestedBuilder.create<affine::AffineStoreOp>(loc, valueToStore, memRef,
                                                    ivs);
      });

  // Replace this operation with the generated alloc.
  rewriter.replaceOp(op, memRef);
}

static void lowerMatmulOpToLoops(Operation *op, ValueRange operands,
                                 PatternRewriter &rewriter,
                                 LoopIterationFn processIteration) {
  auto tensorType = (*op->result_type_begin()).cast<TensorType>();
  auto loc = op->getLoc();
  if (op->getResults().size() != 1) {
    emitError(loc, "Only support operation with 1 result");
  }
  mlir::Value memRef = createStoreOpMemRef(op, rewriter);
  // Determines the lower bound and step size of the loop
  SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value=*/0);
  SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);
  /* Take matrix multiplication A[i,j]xB[j,k]->C[i,k] as an example
     Next, get the dimension hidden in the matrix multiplication process,
    that is, the column j of the first matrix or the row j of the second
    maFtrix */
  auto hiddenDim = op->getOperand(0).getType().cast<ShapedType>().getShape()[1];
  SmallVector<int64_t, 1> dim;
  dim.push_back(hiddenDim);
  /* The outermost layer is a double loop, and the IVS can get the I and K, and
   the outer layer is the I cycle, and the inner layer is the K cycle*/
  affine::buildAffineLoopNest(
      rewriter, loc, lowerBounds, tensorType.getShape(), steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        // Get the i and k of the current loop
        SmallVector<mlir::Value, 3> curIndex;
        curIndex.push_back(ivs[0]);
        curIndex.push_back(ivs[1]);
        ValueRange ResultIndex = ivs;
        // Do an inner loop on the J dimension
        SmallVector<int64_t, 4> lowerBounds(1, /*Value=*/0);
        SmallVector<int64_t, 4> steps(1, /*Value=*/1);
        affine::buildAffineLoopNest(
            rewriter, loc, lowerBounds, dim, steps,
            [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
              // Get the current j
              curIndex.push_back(ivs[0]);
              // Get the result of multiplication
              Value valueAfterMul =
                  processIteration(nestedBuilder, operands, curIndex);
              // accumulate
              auto valueAfterAdd = nestedBuilder.create<affine::AffineLoadOp>(
                  loc, memRef, ResultIndex);
              Value valueToStore = nestedBuilder.create<arith::AddFOp>(
                  loc, valueAfterMul, valueAfterAdd);
              nestedBuilder.create<affine::AffineStoreOp>(loc, valueToStore,
                                                          memRef, ResultIndex);
            });
      });

  // Replace this operation with the generated alloc.
  rewriter.replaceOp(op, memRef);
}

//
// Toy patterns
//

template <typename ToyBinaryOp, typename LoweredBinaryOp>
class ToyBinaryPattern : public ConversionPattern {
public:
  ToyBinaryPattern(MLIRContext *ctx, toy::LoweringPatternMode mode)
      : ConversionPattern(ToyBinaryOp::getOperationName(), 1, ctx), mode(mode) {
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    if (mode == toy::LoweringPatternMode::Vector) {
      mlir::Value memRef = createStoreOpMemRef(op, rewriter);

      const auto tensorType = (*op->result_type_begin()).cast<TensorType>();
      const auto elementType = tensorType.getElementType();
      typename ToyBinaryOp::Adaptor binaryAdaptor(operands);

      // Reshape high dimension memref to 1D memref
      auto lhsInput = binaryAdaptor.getLhs();
      auto lhsType =
          binaryAdaptor.getLhs().getType().template cast<ShapedType>();
      if (lhsType.getRank() > 1) {
        int64_t num = lhsType.getNumElements();
        auto newlhsType = MemRefType::get({num}, lhsType.getElementType());
        std::vector<int64_t> newSize = {num};
        std::vector<int64_t> newStride = {1};
        lhsInput = rewriter.create<memref::ReinterpretCastOp>(
            loc, newlhsType, lhsInput, 0, ArrayRef(newSize),
            ArrayRef(newStride));
      }

      auto rhsInput = binaryAdaptor.getRhs();
      auto rhsType =
          binaryAdaptor.getRhs().getType().template cast<ShapedType>();
      if (rhsType.getRank() > 1) {
        int64_t num = rhsType.getNumElements();
        auto newrhsType = MemRefType::get({num}, rhsType.getElementType());
        std::vector<int64_t> newSize = {num};
        std::vector<int64_t> newStride = {1};
        rhsInput = rewriter.create<memref::ReinterpretCastOp>(
            loc, newrhsType, rhsInput, 0, ArrayRef(newSize),
            ArrayRef(newStride));
      }

      if (tensorType.getRank() > 1) {
        const int64_t length = tensorType.getNumElements();
        auto memRefType = MemRefType::get({length}, elementType);
        std::vector<int64_t> newSize = {length};
        std::vector<int64_t> newStride = {1};
        memRef = rewriter.create<memref::ReinterpretCastOp>(
            loc, memRefType, memRef, 0, ArrayRef(newSize), ArrayRef(newStride));
      }

      // Need to cast buffer according to AVX register number to avoid reload.
      // There are 16 ymms and each of ymm can handle 4 f64, then we can the
      // upper bound should be 64 f64. Here buffer is cut into 16 f64 to save
      // ymm usage.
      const int64_t length = tensorType.getNumElements();
      const int64_t num16f64 = length / 16;
      const int64_t residue = length % 16;

      // Handle 16 f64 (4 ymm) in loops
      // For example, 65 f64 are divided into 4 x 16 + 1
      if (num16f64) {
        const auto loadedEleType = VectorType::get({16}, elementType);
        SmallVector<int64_t, 4> lowerBounds(1, 0);
        SmallVector<int64_t, 4> steps(1, 16);
        SmallVector<int64_t, 4> upperBounds(1, 16 * num16f64);
        affine::buildAffineLoopNest(
            rewriter, loc, lowerBounds, upperBounds, steps,
            [&](OpBuilder &nestedBuilder, Location loc, ValueRange loopIvs) {
              auto loadedVectorLhs = rewriter.create<vector::LoadOp>(
                  loc, loadedEleType, lhsInput, loopIvs);
              auto loadedVectorRhs = rewriter.create<vector::LoadOp>(
                  loc, loadedEleType, rhsInput, loopIvs);
              mlir::Value valueToStore = rewriter.create<LoweredBinaryOp>(
                  loc, loadedVectorLhs, loadedVectorRhs);
              rewriter.create<vector::StoreOp>(loc, valueToStore, memRef,
                                               loopIvs);
            });
      }

      // Handle residue f64 within number 16
      if (residue) {
        SmallVector<int64_t, 4> residueShape;
        residueShape.push_back(residue);
        const auto loadedEleType = VectorType::get(residueShape, elementType);
        auto cst = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIndexType(), rewriter.getIndexAttr(num16f64 * 16));
        SmallVector<mlir::Value, 4> memRefIdx(1, cst);
        auto loadedLhs = rewriter.create<vector::LoadOp>(loc, loadedEleType,
                                                         lhsInput, memRefIdx);
        auto loadedRhs = rewriter.create<vector::LoadOp>(loc, loadedEleType,
                                                         rhsInput, memRefIdx);
        mlir::Value valueToStore =
            rewriter.create<LoweredBinaryOp>(loc, loadedLhs, loadedRhs);
        rewriter.create<vector::StoreOp>(loc, valueToStore, memRef, memRefIdx);
      }

      rewriter.replaceOp(op, memRef);
    } else if (mode == toy::LoweringPatternMode::Loop) {
      lowerOpToLoops(
          op, operands, rewriter,
          [loc](OpBuilder &builder, ValueRange memRefOperands,
                ValueRange loopIvs) {
            // Generate an adaptor for the remapped operands of the
            // BinaryOp. This allows for using the nice named accessors
            // that are generated by the ODS.
            typename ToyBinaryOp::Adaptor binaryAdaptor(memRefOperands);

            // Generate loads for the element of 'lhs' and 'rhs' at the
            // inner loop.
            auto loadedLhs = builder.create<affine::AffineLoadOp>(
                loc, binaryAdaptor.getLhs(), loopIvs);
            auto loadedRhs = builder.create<affine::AffineLoadOp>(
                loc, binaryAdaptor.getRhs(), loopIvs);

            // Create the binary operation performed on the loaded
            // values.
            return builder.create<LoweredBinaryOp>(loc, loadedLhs, loadedRhs);
          });
    } else {
      emitError(loc, "Unsupported lowering binary pattern");
      return mlir::failure();
    }
    return mlir::success();
  }

private:
  enum toy::LoweringPatternMode mode;
};

class ToyMatmulPattern : public OpConversionPattern<toy::MatmulOp> {
  using OpConversionPattern<toy::MatmulOp>::OpConversionPattern;

public:
  ToyMatmulPattern(MLIRContext *ctx, toy::LoweringPatternMode mode)
      : OpConversionPattern(ctx), mode(mode) {}

  LogicalResult
  matchAndRewrite(toy::MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    if (mode == toy::LoweringPatternMode::Vector) {
      mlir::Value memRef = createStoreOpMemRef(op, rewriter);

      const auto tensorType = (*op->result_type_begin()).cast<TensorType>();

      auto cst = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexType(), rewriter.getIndexAttr(0));
      SmallVector<mlir::Value, 4> matIdx(2, cst);

      auto lhsInput = adaptor.getLhs();
      auto lhsType = lhsInput.getType().template cast<ShapedType>();
      const auto lhsVectorType =
          VectorType::get(lhsType.getShape(), lhsType.getElementType());
      auto loadedLhs = rewriter.create<amx::TileLoadOp>(loc, lhsVectorType,
                                                        lhsInput, matIdx);

      auto rhsInput = adaptor.getRhs();
      auto rhsType = rhsInput.getType().template cast<ShapedType>();
      const auto rhsVectorType =
          VectorType::get(rhsType.getShape(), rhsType.getElementType());
      auto loadedRhs = rewriter.create<amx::TileLoadOp>(loc, rhsVectorType,
                                                        rhsInput, matIdx);

      const auto resVectorType =
          VectorType::get(tensorType.getShape(), tensorType.getElementType());
      auto accInput = rewriter.create<amx::TileZeroOp>(loc, resVectorType);

      auto res = rewriter.create<amx::TileMulFOp>(loc, resVectorType, loadedLhs,
                                                  loadedRhs, accInput);

      rewriter.create<amx::TileStoreOp>(loc, memRef, matIdx, res);
      rewriter.replaceOp(op, memRef);
    } else if (mode == toy::LoweringPatternMode::Loop) {
      lowerMatmulOpToLoops(
          op, adaptor.getOperands(), rewriter,
          [loc, &adaptor](OpBuilder &builder, ValueRange memRefOperands,
                          ValueRange loopIvs) {
            // typename toy::MatmulOpAdaptor adaptor(memRefOperands);
            // Get from loopIvs to i, j, k
            SmallVector<mlir::Value, 2> LhsIndex;
            SmallVector<mlir::Value, 2> RhsIndex;
            LhsIndex.push_back(loopIvs[0]);
            LhsIndex.push_back(loopIvs[2]);
            RhsIndex.push_back(loopIvs[1]);
            RhsIndex.push_back(loopIvs[2]);
            // load two numbers to do multiplication
            auto loadedLhs = builder.create<affine::AffineLoadOp>(
                loc, adaptor.getLhs(), LhsIndex);
            auto loadedRhs = builder.create<affine::AffineLoadOp>(
                loc, adaptor.getRhs(), RhsIndex);
            return builder.create<arith::MulFOp>(loc, loadedLhs, loadedRhs);
          });
    } else {
      emitError(loc, "Unsupported lowering binary pattern");
      return mlir::failure();
    }
    return mlir::success();
  }

private:
  enum toy::LoweringPatternMode mode;
};

using ToyAddPattern = ToyBinaryPattern<toy::AddOp, arith::AddFOp>;
using ToyMulPattern = ToyBinaryPattern<toy::MulOp, arith::MulFOp>;

class ToyConstantOpPattern : public OpConversionPattern<toy::ConstantOp> {
  using OpConversionPattern<toy::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    DenseElementsAttr constantValue = op.getValue();
    Location loc = op.getLoc();

    // When lowering the constant operation, we allocate and assign the constant
    // values to a corresponding memref allocation.
    auto tensorType = op.getType().cast<TensorType>();
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // We will be generating constant indices up-to the largest dimension.
    // Create these constants up-front to avoid large amounts of redundant
    // operations.
    auto valueShape = memRefType.getShape();
    SmallVector<Value, 8> constantIndices;

    if (!valueShape.empty()) {
      for (auto i : llvm::seq<int64_t>(
               0, *std::max_element(valueShape.begin(), valueShape.end())))
        constantIndices.push_back(
            rewriter.create<arith::ConstantIndexOp>(loc, i));
    } else {
      // This is the case of a tensor of rank 0.
      constantIndices.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, 0));
    }

    // The constant operation represents a multi-dimensional constant, so we
    // will need to generate a store for each of the elements. The following
    // functor recursively walks the dimensions of the constant shape,
    // generating a store when the recursion hits the base case.
    SmallVector<Value, 2> indices;
    auto valueIt = constantValue.value_begin<FloatAttr>();
    std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
      // The last dimension is the base case of the recursion, at this point
      // we store the element at the given index.
      if (dimension == valueShape.size()) {
        rewriter.create<affine::AffineStoreOp>(
            loc, rewriter.create<arith::ConstantOp>(loc, *valueIt++), alloc,
            llvm::ArrayRef(indices));
        return;
      }

      // Otherwise, iterate over the current dimension and add the indices to
      // the list.
      for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
        indices.push_back(constantIndices[i]);
        storeElements(dimension + 1);
        indices.pop_back();
      }
    };

    // Start the element storing recursion from the first dimension.
    storeElements(/*dimension=*/0);

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

class ToyFuncOpPattern : public OpConversionPattern<toy::FuncOp> {
  using OpConversionPattern<toy::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Convert arguments type from tensor to memref
    const auto args = op.getRegion().getArguments();
    llvm::SmallVector<mlir::Type, 4> argTypes;
    for (auto &a : args) {
      auto tensorType = a.getType().cast<TensorType>();
      auto memRefType = convertTensorToMemRef(tensorType);
      a.setType(memRefType);
      argTypes.push_back(memRefType);
    }

    // Convert results type from tensor to memref
    llvm::SmallVector<mlir::Type, 4> resTypes;
    for (auto &opi : op.getBody().getOps()) {
      if (auto retOp = mlir::dyn_cast<toy::ReturnOp>(opi)) {
        for (auto t : retOp.getInput().getType()) {
          auto tensorType = t.cast<TensorType>();
          auto memRefType = convertTensorToMemRef(tensorType);
          resTypes.push_back(memRefType);
        }
      }
    }

    // Create a new non-toy function, with the same region.
    auto newFunctionType = rewriter.getFunctionType(argTypes, resTypes);
    auto func = rewriter.create<mlir::func::FuncOp>(op.getLoc(), op.getName(),
                                                    newFunctionType);

    rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
    rewriter.eraseOp(op);
    return success();
  }
};

class ToyReturnOpPattern : public OpConversionPattern<toy::ReturnOp> {
  using OpConversionPattern<toy::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // We lower "toy.return" directly to "func.return".
    // Real time info can be got from adaptor.
    // op.getInput only gets origin info.
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getInput());
    return success();
  }
};

class ToyPrintOpPattern : public OpConversionPattern<toy::PrintOp> {
  using OpConversionPattern<toy::PrintOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(toy::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // We don't lower "toy.print" in this pass, but we need to update its
    // operands.
    rewriter.modifyOpInPlace(op,
                             [&] { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

//
// Targets and patterns definition and registration
//

class ConvertToyToMid : public ConvertToyToMidBase<ConvertToyToMid> {
public:
  ConvertToyToMid() = default;
  ConvertToyToMid(mlir::toy::LoweringPatternMode mode) {
    loweringPatternMode = mode;
  };

  void runOnOperation() override {
    // The first thing to define is the conversion target. This will define the
    // final target for this lowering.
    ConversionTarget target(getContext());

    // We define the specific operations, or dialects, that are legal targets
    // for this lowering. In our case, we are lowering to a combination of the
    // `Affine`, `Arith`, `Func`, and `MemRef` dialects.
    target.addLegalDialect<affine::AffineDialect, BuiltinDialect,
                           arith::ArithDialect, func::FuncDialect,
                           memref::MemRefDialect, vector::VectorDialect,
                           amx::AMXDialect>();

    // We also define the Toy dialect as Illegal so that the conversion will
    // fail if any of these operations are *not* converted. Given that we
    // actually want a partial lowering, we explicitly mark the Toy operations
    // that don't want to lower, `toy.print`, as `legal`. `toy.print` will still
    // need its operands to be updated though (as we convert from TensorType to
    // MemRefType), so we only treat it as `legal` if its operands are legal.
    target.addIllegalDialect<toy::ToyDialect>();
    target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op) {
      return llvm::none_of(op->getOperandTypes(),
                           [](Type type) { return type.isa<TensorType>(); });
    });

    // Now that the conversion target has been defined, we just need to provide
    // the set of patterns that will lower the Toy operations.
    RewritePatternSet patterns(&getContext());
    patterns.add<ToyFuncOpPattern, ToyReturnOpPattern, ToyPrintOpPattern,
                 ToyConstantOpPattern>(&getContext());
    switch (loweringPatternMode) {
    case toy::LoweringPatternMode::Loop:
      patterns.add<ToyAddPattern, ToyMulPattern, ToyMatmulPattern>(
          &getContext(), toy::LoweringPatternMode::Loop);
      break;
    case toy::LoweringPatternMode::Vector:
      patterns.add<ToyAddPattern, ToyMulPattern, ToyMatmulPattern>(
          &getContext(), toy::LoweringPatternMode::Vector);
      break;
    default:
      llvm::errs() << "Unsupported lowering binary pattern\n";
      signalPassFailure();
    }

    // With the target and rewrite patterns defined, we can now attempt the
    // conversion. The conversion will signal failure if any of our `illegal`
    // operations were not converted successfully.
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::toy::createConvertToyToMidPass() {
  return std::make_unique<ConvertToyToMid>();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::toy::createConvertToyToMidPass(mlir::toy::LoweringPatternMode mode) {
  return std::make_unique<ConvertToyToMid>(mode);
}
