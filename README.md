# The Toy Compiler

This project is an implementation of compiler using
the [Multi-Level Intermediate Representation](https://mlir.llvm.org/) (MLIR)
framework. In the first stage, project is constructed by using the MLIR
Toy [examples](https://github.com/llvm/llvm-project/tree/main/mlir/docs/Tutorials/Toy), and it can be seen as a
reconstruction of Toy. Therefore the name "Toy Compiler" is adopted. Here, there are several improvements during the
reconstruction:

* MLIR/LLVM project is added as a third party package using `find_package` in the CMakeLists. This makes Toy
  Compiler an independent project but not joinly built with MLIR.
* The project directories are re-designed by taking [Triton](https://github.com/openai/triton) as a reference. This
  makes
  the project structure much clearer and easier for further extension.
* .gitignore and .clang-format are added to make the Toy from the tutorial to a real project.

## Getting Started with the Toy Compiler

### How to build the project

Clone the project.

```bash
git clone https://github.com/LeiChenIntel/toy-compiler.git
```

Move to Toy Compiler based directory and update submodule. Now the project depends on GoogleTest.

```bash
cd $TOY_COMPILER_HOME_DIR
git submodule update --init
```

Build MLIR framework by following [MLIR Getting Started](https://mlir.llvm.org/getting_started/).

The branch `release/18.x` is used. Note that using different branches might cause errors.

More details can be found in the [checklist](docs/checklist.md).

For Linux platform, build the project by

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

For Windows platform, build the project by

```bash
mkdir $TOY_COMPILER_HOME_DIR/build-x86_64
cd $TOY_COMPILER_HOME_DIR/build-x86_64
cmake -G "Visual Studio 16 2019" -A x64 -D CMAKE_PREFIX_PATH="$LLVM_HOME_DIR/build/lib/cmake/mlir" -D CMAKE_BUILD_TYPE=Release ..
cmake --build . --target install
```

Other CMake options:

`-DENABLE_TOY_BENCHMARKS`: Enable benchmarks. Compare the results with loop and AVX instructions. Need AVX support and
release build type. Default value ON.

`-DENABLE_MATMUL_BENCHMARKS`: Enable matrix multiplication benchmark. Need to install OpenBLAS library. Only supported
on Ubuntu. Default value OFF.

### How to run the target

#### toy-opt

**toy-opt** is designed to test single pass lit. File name for IR and pass name are necessary to run toy-opt.

Here is an example to run ConvertToyToMid pass by toy-opy,

```bash
./toy-opt toy_to_mid.mlir -convert-toy-to-mid
```

More usage can be found by running,

```bash
./toy-opt -h
```

#### toy-translate

**toy-translate** is designed to translate one IR to another IR. Basically, toy-translate is a pass pipeline and may
include many passes.

Options:

-emit: Choose an IR to be dumped, i.e., ast, mlir, mlir-mid.

-opt: If optimization pass is added, such as canonicalization and common sub-expressions elimination.

-lower-pat: Choose loop or vectorization during the lowering, i.e., loop, vector.

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

Then the expected outputs like as

```bash
[4/5] Running the toy lit tests

Testing Time: 0.04s
  Passed: 2
```

#### Unit test

Unit test based on GoogleTest framework is applied in this project. `unit-test` can be run directly, and the output
likes as

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

More details can be found on page [benchmarks](docs/benchmarks.md)
