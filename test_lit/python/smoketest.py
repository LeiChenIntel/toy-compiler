import sys
from mlir_toy.ir import *
# This directory is related to PYTHONPATH=$TOY_COMPILER_BINARY_DIR/python_packages/toy
# mlir_toy starts from PYTHONPATH
from mlir_toy.dialects import builtin as builtin_d

if sys.argv[1] == "pybind11":
    from mlir_toy.dialects import toy_pybind11 as toy_d
elif sys.argv[1] == "nanobind":
    from mlir_toy.dialects import toy_nanobind as toy_d
else:
    raise ValueError("Expected either pybind11 or nanobind as arguments")

with Context() as ctx, Location.unknown():
    toy_d.register_dialect()
    module1 = Module.parse(
        """
            %100 = arith.constant 2 : i32
            %0 = toy.constant {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>} : tensor<6xf64>
            %1 = toy.reshape(%0) : tensor<6xf64> -> tensor<2x3xf64>
            %2 = toy.constant {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>} : tensor<6xf64>
            %3 = toy.reshape(%2) : tensor<6xf64> -> tensor<2x3xf64>
            %4 = toy.add(%1, %3) : tensor<2x3xf64>, tensor<2x3xf64> -> tensor<2x3xf64>
            toy.print(%4) : tensor<2x3xf64>
        """
    )
    print(str(module1))
    # Module can be parsed from a string, and printed as
    # module {
    #     %c2_i32 = arith.constant 2 : i32
    #     %0 = toy.constant {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>} : tensor<6xf64>
    #     %1 = toy.reshape(%0) : tensor<6xf64> -> tensor<2x3xf64>
    #     %2 = toy.constant {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>} : tensor<6xf64>
    #     %3 = toy.reshape(%2) : tensor<6xf64> -> tensor<2x3xf64>
    #     %4 = toy.add(%1, %3) : tensor<2x3xf64>, tensor<2x3xf64> -> tensor<2x3xf64>
    #     toy.print(%4) : tensor<2x3xf64>
    # }

    module2 = Module.create()
    with InsertionPoint(module2.body):
        # Find the class type in _toy_ops_gen.py

        # input_types = RankedTensorType.get([2, 3], F64Type.get())
        # output_types = RankedTensorType.get([2, 3], F64Type.get())
        # function_type2 = FunctionType.get(inputs=[input_types], results=[output_types])

        function_type = FunctionType.get([], [])
        # Convert to TypeAttr for the operation
        function_type_attr = TypeAttr.get(function_type)
        f = toy_d.FuncOp(sym_name="add_constant_fold", function_type=function_type_attr)

        body_region = f.regions[0]
        # Create a block at the start of the region
        # The argument types should match the function's input parameter types
        block = Block.create_at_start(body_region, [])
        with InsertionPoint(block):
            constant_op1 = toy_d.ConstantOp(
                result=RankedTensorType.get([6], F64Type.get()),
                value=DenseElementsAttr.get_splat(
                    RankedTensorType.get([6], F64Type.get()),
                    FloatAttr.get(F64Type.get(), 1.0)))
            reshape_op1 = toy_d.ReshapeOp(
                input=constant_op1.result,
                output=RankedTensorType.get([2, 3], F64Type.get()))

            constant_op2 = toy_d.ConstantOp(
                result=RankedTensorType.get([6], F64Type.get()),
                value=DenseElementsAttr.get_splat(
                    RankedTensorType.get([6], F64Type.get()),
                    FloatAttr.get(F64Type.get(), 1.0)))
            reshape_op2 = toy_d.ReshapeOp(
                input=constant_op2.result,
                output=RankedTensorType.get([2, 3], F64Type.get()))

            add_op = toy_d.AddOp(lhs=reshape_op1.result, rhs=reshape_op2.result)

            print_op = toy_d.PrintOp(input=add_op.result)

            toy_d.ReturnOp([])

    print(str(module2))
    # Module can be constructed programmatically, and printed as
    # module {
    #     toy.func @add_constant_fold() {
    #         %0 = toy.constant {value = dense<1.000000e+00> : tensor<6xf64>} : tensor<6xf64>
    #         %1 = toy.reshape(%0) : tensor<6xf64> -> tensor<2x3xf64>
    #         %2 = toy.constant {value = dense<1.000000e+00> : tensor<6xf64>} : tensor<6xf64>
    #         %3 = toy.reshape(%2) : tensor<6xf64> -> tensor<2x3xf64>
    #         %4 = toy.add(%1, %3) : tensor<2x3xf64>, tensor<2x3xf64> -> tensor<2x3xf64>
    #         toy.print(%4) : tensor<2x3xf64>
    #         toy.return
    #     }
    # }
