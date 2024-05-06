set(LLVM_TARGET_DEFINITIONS ToyDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS ToyOps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
add_public_tablegen_target(ToyOpsIncGen)

set(LLVM_TARGET_DEFINITIONS ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(ToyCombineIncGen)

set(LLVM_TARGET_DEFINITIONS ToyInterface.td)
mlir_tablegen(ToyInterface.h.inc -gen-op-interface-decls)
mlir_tablegen(ToyInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(ToyInterfaceIncGen)
