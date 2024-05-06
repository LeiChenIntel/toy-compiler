#ifndef TOY_INTERFACE_TD
#define TOY_INTERFACE_TD

include "mlir/IR/OpBase.td"

def ToyOpInterface : OpInterface<"ToyOpInterface"> {
  let description = [{
      This is an example.
  }];

  let methods = [
    InterfaceMethod<"Infer byte size cost of each operation.",
                    "int", "giveInputOutputByteSize">
  ];
}

#endif // TOY_INTERFACE_TD
