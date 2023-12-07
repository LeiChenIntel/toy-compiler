# Benchmarks

##### 1. Add tensors

Add tensor is the "Hello World!" in code optimization. This benchmark includes simple loop addition and vector addition
using AVX2 instruction. A Toy language file is compiled to LLVM IR, and then to x86 instruction in backend.

Note that this benchmark needs AVX2 support in the target machine and build type is set to Release.

It can be run directly by

```bash
benchmark-add-tensors.exe
```
