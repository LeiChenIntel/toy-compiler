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

with Context():
    toy_d.register_dialect()
