# The Toy Compiler

This project is an implementation of compiler using 
[Multi-Level Intermediate Representation](https://mlir.llvm.org/) (MLIR)
framework. In the first stage, the project is constructed by using the MLIR
Toy [examples](https://github.com/llvm/llvm-project/tree/main/mlir/docs/Tutorials/Toy), and it can be seen as a
reconstruction of Toy. Therefore, the name "Toy Compiler" is adopted. There are several improvements during the
reconstruction:

* The MLIR/LLVM project is added as a third-party package using `find_package` in the CMakeLists. This makes Toy
  Compiler an independent project but not jointly built with MLIR.
* The project directories are redesigned by taking [Triton](https://github.com/openai/triton) as a reference. This
  makes the project structure much clearer and easier for further extension.
* .gitignore and .clang-format are added to transform the Toy tutorial into a real project.

## Toy Language

The Toy language is a simple interface to create Toy IR using Python. Developers will not need to work with C++ code 
or write a Toy IR manually.

Here is an example of creating a Toy ConstantOp in the module.
```python
constant_op1 = toy_d.ConstantOp(
    result=RankedTensorType.get([6], F64Type.get()),
    value=DenseElementsAttr.get_splat(
        RankedTensorType.get([6], F64Type.get()),
        FloatAttr.get(F64Type.get(), 1.0)))
```
The module can then be dumped as
```text
module {
  %0 = toy.constant {value = dense<1.000000e+00> : tensor<6xf64>} : tensor<6xf64>
}
```
More details can be found on page [Python Binding](docs/python-binding.md).

## Getting Started with the Toy Compiler

### How to build the project

Clone the project.

```bash
git clone https://github.com/LeiChenIntel/mlir-toy-compiler.git
```

Move to the Toy Compiler base directory and update submodule. The project now depends on GoogleTest.

```bash
cd $TOY_COMPILER_HOME_DIR
git submodule update --init
```

Build the MLIR framework by following [MLIR Getting Started](https://mlir.llvm.org/getting_started/).

The branch `release/21.x` (`2078da43`) is used. Note that using different branches might cause errors.

More details can be found in the [checklist](docs/checklist.md).

For **Ubuntu** platform, build the project by

```bash
mkdir $TOY_COMPILER_HOME_DIR/build-x86_64
cd $TOY_COMPILER_HOME_DIR/build-x86_64
cmake -D CMAKE_PREFIX_PATH=$LLVM_HOME_DIR/build/lib/cmake/mlir -D CMAKE_BUILD_TYPE=Release ..
make -j${nproc}
```

* MLIRConfig.cmake should be under path `$LLVM_HOME_DIR/build/lib/cmake/mlir`, or there will be an error during CMake
  generation.
* If you want to put generated binaries under folder bin and lib, run `cmake --build . --target install` instead
  of `make -j${nproc}`.

⚠️ The development targets to Ubuntu now. Might have issue in Windows build.

For Windows platform, build the project by

```bash
mkdir $TOY_COMPILER_HOME_DIR/build-x86_64
cd $TOY_COMPILER_HOME_DIR/build-x86_64
cmake -G "Visual Studio 16 2019" -A x64 -D CMAKE_PREFIX_PATH="$LLVM_HOME_DIR/build/lib/cmake/mlir" -D CMAKE_BUILD_TYPE=Release ..
cmake --build . --target install
```

Additional CMake options:

`-DENABLE_TOY_BENCHMARKS`: Enable benchmarks. Compare results with loop and AVX instructions. Requires AVX support and 
release build type. Default value ON.

`-DENABLE_MATMUL_BENCHMARKS`: Enable matrix multiplication benchmark. Requires OpenBLAS library installation.
Only supported on Ubuntu. Default value OFF.

### How to run the target

#### toy-opt

**toy-opt** is designed to test single pass lit. IR filename and pass name are required to run toy-opt.

Here is an example of running the ConvertToyToMid pass with toy-opt

```bash
./toy-opt toy_to_mid.mlir -convert-toy-to-mid
```

More usage can be found by running,

```bash
./toy-opt -h
```

#### toy-translate

**toy-translate** is designed to translate IR from one form to another.
Essentially, toy-translate is a pass pipeline and may include many passes.

Options:

-emit: Choose which IR to dump, i.e., ast, mlir, mlir-mid.

-opt: Whether optimization passes are added, such as canonicalization and common sub-expressions elimination.

-lower-pat: Choose between loop or vectorization during lowering, i.e., loop, vector.

Here is an example to dump MidIR by toy-translate,

```bash
./toy-translate add_mlir.toy -emit=mlir-mid -opt
```

#### LIT test

LIT test is a test framework in MLIR which can help to track the differences of each IR after single or multiple passes.

```bash
cd $TOY_COMPILER_HOME_DIR/build-x86_64
cmake --build . --config Release --target check-toy-lit
```

The expected output should look like:

```bash
[4/5] Running the toy lit tests

Testing Time: 0.04s
  Passed: 2
```

#### Unit test

Unit tests based on the GoogleTest framework is applied in this project. unit-test can be run directly,
and the output looks like:

```bash
[==========] Running 1 test from 1 test suite.
[----------] Global test environment set-up.
[----------] 1 test from MLIR
[ RUN      ] MLIR.BuiltinTypes
[       OK ] MLIR.BuiltinTypes (0 ms)
[----------] 1 test from MLIR (0 ms total)

[----------] Global test environment tear-down
[==========] 1 test from 1 test suite ran. (0 ms total)
[  PASSED  ] 1 test.
```

#### Examples

More details can be found on page [examples](docs/examples.md).

#### Benchmarks

More details can be found on page [benchmarks](docs/benchmarks.md).
