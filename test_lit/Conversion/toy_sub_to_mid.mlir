// RUN: toy-opt %s -convert-toy-sub-to-mid | FileCheck %s

// CHECK-LABEL: @convert_sub_to_mid
toy.func @convert_sub_to_mid(%arg0: memref<2x3xf64>, %arg1: memref<2x3xf64>) {
    %0 = toy.sub(%arg0, %arg1) : memref<2x3xf64>, memref<2x3xf64> -> tensor<2x3xf64>
    toy.print(%0) : tensor<2x3xf64>
    toy.return

    // CHECK: %alloc = memref.alloc() : memref<2x3xf64>
    // CHECK: affine.for %arg2 = 0 to 2 {
    // CHECK: affine.for %arg3 = 0 to 3 {
    // CHECK: %0 = affine.load %arg0[%arg2, %arg3] : memref<2x3xf64>
    // CHECK: %1 = affine.load %arg1[%arg2, %arg3] : memref<2x3xf64>
    // CHECK: %2 = arith.subf %0, %1 : f64
    // CHECK: affine.store %2, %alloc[%arg2, %arg3] : memref<2x3xf64>
    // CHECK: toy.print(%alloc) : memref<2x3xf64>
    // CHECK: memref.dealloc %alloc : memref<2x3xf64>
    // CHECK: return
}
