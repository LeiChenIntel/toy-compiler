#ifndef TOY_COMBINE_TD
#define TOY_COMBINE_TD

include "mlir/IR/PatternBase.td"
include "Toy/ToyOps.td"

//
// ReshapeOp
//

// Reshape(Reshape(x)) = Reshape(x)
def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)),
                                   (ReshapeOp $arg)>;

// Reshape(Constant(x)) = x'
def ReshapeConstant :
    NativeCodeCall<"$0.reshape(($1.getType()).cast<ShapedType>())">;
def FoldConstantReshapeOptPattern : Pat<
    (ReshapeOp:$res (ConstantOp $arg)),
    (ConstantOp (ReshapeConstant $arg, $res))>;

// Reshape(x) = x, where input and output shapes are identical
def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;
def RedundantReshapeOptPattern : Pat<
    (ReshapeOp:$res $arg), (replaceWithValue $arg),
    [(TypesAreIdentical $res, $arg)]>;

#endif // TOY_COMBINE_TD
