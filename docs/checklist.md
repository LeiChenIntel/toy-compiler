# Checklist

The check list is a record for issues during development and deployment.

##### 1. build llvm-project on Windows platform
In the official documents, llvm-project is built by using Visual Studio 2017. This is a bit out of date. Visual Studio 
2019 compiler is applied in this project and the commands for llvm-project are
```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout release/16.x
mkdir build && cd build
cmake ..\llvm -G "Visual Studio 16 2019" -A x64 -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BU
ILD="host" -DCMAKE_BUILD_TYPE=Release -Thost=x64 -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON
cmake --build . --config Release --target tools/mlir/test/check-mlir
```

##### 2. not recognized as an internal or external command
Build type of toy-compiler and llvm-project should be the same. Or try to rename `Release` to `RelWithDebInfo`. It works
in my env.

##### 3. llvm-lit.py path does not exist
```bash
LLVM_EXTERNAL_LIT set to
../llvm-project/build/$(Configuration)/bin/llvm-lit.py, but the
path does not exist.
```
Need to check path `build/$(Configuration)`. Copy or rename `build/Release` to `build/$(Configuration)`.
