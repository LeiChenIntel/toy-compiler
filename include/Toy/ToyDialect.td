#ifndef TOY_DIALECT_TD
#define TOY_DIALECT_TD

include "mlir/IR/OpBase.td"

def Toy_Dialect : Dialect {
  let name = "toy";

  let cppNamespace = "::mlir::toy";

  let summary = "A high-level dialect for analyzing and optimizing the Toy language";

  let description = [{
      Toy Dialect.
  }];
}

#endif // TOY_DIALECT_TD
