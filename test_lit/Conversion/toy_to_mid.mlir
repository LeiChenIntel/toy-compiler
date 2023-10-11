// RUN: toy-opt %s -convert-toy-to-mid | FileCheck %s

// CHECK-LABEL: @convert_print_to_mid
toy.func @convert_print_to_mid(%arg0: tensor<2x3xf64>) {
    toy.print(%arg0) : tensor<2x3xf64>
    toy.return

    // CHECK: [[VAL_0:%.*]] = memref.alloc() : memref<2x3xf64>
    // CHECK: toy.print([[VAL_0]]) : memref<2x3xf64>
    // CHECK: memref.dealloc [[VAL_0]] : memref<2x3xf64>
    // CHECK: return
}

// CHECK-LABEL: @convert_add_to_mid
toy.func @convert_add_to_mid(%arg0: tensor<2x3xf64>, %arg1: tensor<2x3xf64>) {
    %0 = toy.add(%arg0, %arg1) : tensor<2x3xf64>, tensor<2x3xf64> -> tensor<2x3xf64>
    toy.print(%0) : tensor<2x3xf64>
    toy.return

    // CHECK: %alloc = memref.alloc() : memref<2x3xf64>
    // CHECK: %alloc_0 = memref.alloc() : memref<2x3xf64>
    // CHECK: %alloc_1 = memref.alloc() : memref<2x3xf64>
    // CHECK: affine.for %arg2 = 0 to 2 {
    // CHECK: affine.for %arg3 = 0 to 3 {
    // CHECK: %0 = affine.load %alloc_1[%arg2, %arg3] : memref<2x3xf64>
    // CHECK: %1 = affine.load %alloc_0[%arg2, %arg3] : memref<2x3xf64>
    // CHECK: %2 = arith.addf %0, %1 : f64
    // CHECK: affine.store %2, %alloc[%arg2, %arg3] : memref<2x3xf64>
    // CHECK: toy.print(%alloc) : memref<2x3xf64>
    // CHECK: memref.dealloc %alloc_1 : memref<2x3xf64>
    // CHECK: memref.dealloc %alloc_0 : memref<2x3xf64>
    // CHECK: memref.dealloc %alloc : memref<2x3xf64>
    // CHECK: return
}
