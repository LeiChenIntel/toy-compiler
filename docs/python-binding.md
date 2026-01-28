# Python Binding

##### 1. Setup the environment
Create python virtual environment and install the required packages.
```bash
virtualenv mlirdev
source mlirdev/bin/activate
pip install --upgrade pip
cd $LLVM_PROJECT_DIR
pip install -r mlir/python/requirements.txt
```
Rebuild LLVM and MLIR with Python bindings `MLIR_ENABLE_BINDINGS_PYTHON` enabled.
```bash
mkdir build && cd build
cmake -G Ninja ../llvm -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="Native" -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON -DMLIR_ENABLE_BINDINGS_PYTHON=ON
cmake --build . --target check-mlir
```
Create a new build directory under llvm-project if it triggers issues as
```text
CMake Error at /home/leichen1/develop/toy/llvm-project/mlir/cmake/modules/MLIRDetectPythonEnv.cmake:50 (find_package):
  Could not find a package configuration file provided by "pybind11"
  (requested version 2.10) with any of the following names:
```
Validate the installation by running tests and python scripts.
```bash
ninja check-mlir-python
```
logs:
```text
[0/1] Running lit suite /home/leichen1/develop/toy/llvm-project/mlir/test/python
Testing Time: 2.99s
Total Discovered Tests: 90
  Unsupported:  5 (5.56%)
  Passed     : 85 (94.44%)
```
Set up `PYTHONPATH` to include the MLIR python bindings.
```bash
export PYTHONPATH=$LLVM_PROJECT_DIR/build/tools/mlir/python_packages/mlir_core
cd $LLVM_PROJECT_DIR/mlir/test/python/dialects
python arith_dialect.py
```
logs:
```text
TEST: testConstantOps
module {
  %cst = arith.constant 4.242000e+01 : f32
}
...
TEST: testFastMathFlags
%0 = arith.addf %cst, %cst fastmath<nnan,ninf> : f3
```

##### 2. Build Toy-Compiler with Python binding
Set `ENABLE_PYTHON_BINDINGS=ON` and rebuild toy-compiler.

##### 3. Run Toy-Compiler Python binding examples
Check command line is under proper python virtual environment.

Reset `PYTHONPATH` to include toy-compiler python bindings. Example,
```bash
export PYTHONPATH=$TOY_COMPILER_BINARY_DIR/python_packages/toy
[mlir-toy-compiler/cmake-build-release/python_packages/toy]
```
Run python script under directory `mlir-toy-compiler/test_lit/python`
```bash
cd mlir-toy-compiler/test_lit/python
python smoketest.py pybind11
```
