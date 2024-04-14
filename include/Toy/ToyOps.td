#ifndef TOY_OPS_TD
#define TOY_OPS_TD

include "Toy/ToyDialect.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Toy_Op<string mnemonic, list<Trait> traits = []> :
        Op<Toy_Dialect, mnemonic, traits>;

//
// ConstantOp
//

def ConstantOp : Toy_Op<"constant", [Pure]> {
  let summary = "constant";

  let description = [{
      Toy Constant Operation.
  }];

  let arguments = (ins F64ElementsAttr:$value);

  let results = (outs F64Tensor:$result);

  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

//
// ReshapeOp
//

def ReshapeOp : Toy_Op<"reshape", [Pure]> {
  let summary = "reshape";

  let description = [{
      Toy Reshape Operation.
  }];

  let arguments = (ins AnyRankedTensor:$input);

  // We expect that the reshape operation returns a statically shaped tensor.
  let results = (outs StaticShapeTensorOf<[AnyType]>:$output);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` type(operands) `->` type(results)
  }];

  // Enable this flag to add patterns in getCanonicalizationPatterns
  let hasCanonicalizer = 1;
  // Enable this flag to check shape, type of an operator
  let hasVerifier = 1;
}

//
// AddOp
//

def AddOp : Toy_Op<"add", [
  Pure,
  InferTypeOpInterface,
  DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>
]> {
  let summary = "add";

  let description = [{
      Toy Add Operation.
  }];

  let arguments = (ins AnyTypeOf<[AnyMemRef, AnyTensor]>:$lhs, AnyTypeOf<[AnyMemRef, AnyTensor]>:$rhs);
  let results = (outs AnyTypeOf<[AnyMemRef, AnyTensor]>:$output);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` type(operands) `->` type(results)
  }];

  // Set the folder bit so that we can implement constant folders.
  let hasFolder = 1;
  let hasVerifier = 1;
}

//
// MatmulOp
//

def MatmulOp : Toy_Op<"matmul", [
  Pure,
  InferTypeOpInterface,
  DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>
  ]>{
  let summary = "matrix multiplication";

  let description = [{
      Toy matrix multiplication.
  }];

  let arguments = (ins AnyTypeOf<[AnyMemRef, AnyTensor]>:$lhs, AnyTypeOf<[AnyMemRef, AnyTensor]>:$rhs);
  let results = (outs AnyTypeOf<[AnyMemRef, AnyTensor]>:$output);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` type(operands) `->` type(results)
  }];
  let hasFolder = 1;
  let hasVerifier = 1;
}

//
// MulOp
//

def MulOp : Toy_Op<"mul", [
  Pure,
  InferTypeOpInterface,
  DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>
]> {
  let summary = "element-wise multiplication";

  let description = [{
      Toy Element-wise Multiplication Operation.
  }];

  let arguments = (ins AnyTypeOf<[AnyMemRef, AnyTensor]>:$lhs, AnyTypeOf<[AnyMemRef, AnyTensor]>:$rhs);
  let results = (outs AnyTypeOf<[AnyMemRef, AnyTensor]>:$output);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` type(operands) `->` type(results)
  }];

  let hasFolder = 1;
  let hasVerifier = 1;
}

//
// StoreOp
//

def StoreOp : Toy_Op<"store"> {
  let summary = "store";

  let description = [{
      Toy Store Operation.
  }];

  let arguments = (ins AnyTypeOf<[AnyMemRef, AnyRankedTensor]>:$valToStore, AnyTypeOf<[AnyMemRef, AnyRankedTensor]>:$memref);

  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` type(operands)
  }];
}

//
// PrintOp
//

def PrintOp : Toy_Op<"print"> {
  let summary = "print";

  let description = [{
      Toy Print Operation.
  }];

  // In lowering IR, toy.print has MemRef inputs
  let arguments = (ins AnyTypeOf<[AnyMemRef, AnyTensor]>:$input);

  let assemblyFormat = [{
    `(` $input `)` attr-dict `:` type($input)
  }];
}

//
// FuncOp
//

def FuncOp : Toy_Op<"func", [
    FunctionOpInterface, IsolatedFromAbove
    // Declare "IsolatedFromAbove" to create an independent region
  ]> {
  let summary = "user defined function";

  let description = [{
      Toy Function Operation.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs);
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
  >];

  // getArgumentTypes and getResultTypes should be defined in FunctionOpInterface
  // getCallableRegion is needed in CallInterfaces
  let extraClassDeclaration = [{
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
    Region *getCallableRegion() { return &getBody(); }
  }];
  // If 'hasCustomAssemblyFormat' is set as ture, need to define print/parse function
  // in the source file. Or define 'assemblyFormat' to format IR.
  let hasCustomAssemblyFormat = 1;
}

//
// ReturnOp
//

def ReturnOp : Toy_Op<"return", [
    Pure, HasParent<"FuncOp">, Terminator
    // Declare "Terminator" for the last operation
  ]> {
  let summary = "return";

  let description = [{
      Toy Return Operation.
  }];

  let arguments = (ins Variadic<AnyTypeOf<[AnyMemRef, AnyRankedTensor]>>:$input);

  let assemblyFormat = [{
    (`(` $input^ `)` type($input))? attr-dict
  }];
}

#endif // TOY_OPS_TD
