// RUN: toy-opt %s -convert-toy-to-mid | FileCheck %s

// CHECK-LABEL: @convert_print_to_mid
toy.func @convert_print_to_mid(%arg0: tensor<2x3xf64>) {
    toy.print(%arg0) : tensor<2x3xf64>
    toy.return

    // CHECK: toy.print(%arg0) : memref<2x3xf64>
    // CHECK: return
}

// CHECK-LABEL: @convert_add_to_mid
toy.func @convert_add_to_mid(%arg0: tensor<2x3xf64>, %arg1: tensor<2x3xf64>) {
    %0 = toy.add(%arg0, %arg1) : tensor<2x3xf64>, tensor<2x3xf64> -> tensor<2x3xf64>
    toy.print(%0) : tensor<2x3xf64>
    toy.return

    // CHECK: %alloc = memref.alloc() : memref<2x3xf64>
    // CHECK: affine.for %arg2 = 0 to 2 {
    // CHECK: affine.for %arg3 = 0 to 3 {
    // CHECK: %0 = affine.load %arg0[%arg2, %arg3] : memref<2x3xf64>
    // CHECK: %1 = affine.load %arg1[%arg2, %arg3] : memref<2x3xf64>
    // CHECK: %2 = arith.addf %0, %1 : f64
    // CHECK: affine.store %2, %alloc[%arg2, %arg3] : memref<2x3xf64>
    // CHECK: toy.print(%alloc) : memref<2x3xf64>
    // CHECK: memref.dealloc %alloc : memref<2x3xf64>
    // CHECK: return
}

// CHECK-LABEL: @convert_mul_to_mid
toy.func @convert_mul_to_mid(%arg0: tensor<2x3xf64>, %arg1: tensor<2x3xf64>) {
    %0 = toy.mul(%arg0, %arg1) : tensor<2x3xf64>, tensor<2x3xf64> -> tensor<2x3xf64>
    toy.print(%0) : tensor<2x3xf64>
    toy.return

    // CHECK: %alloc = memref.alloc() : memref<2x3xf64>
    // CHECK: affine.for %arg2 = 0 to 2 {
    // CHECK: affine.for %arg3 = 0 to 3 {
    // CHECK: %0 = affine.load %arg0[%arg2, %arg3] : memref<2x3xf64>
    // CHECK: %1 = affine.load %arg1[%arg2, %arg3] : memref<2x3xf64>
    // CHECK: %2 = arith.mulf %0, %1 : f64
    // CHECK: affine.store %2, %alloc[%arg2, %arg3] : memref<2x3xf64>
    // CHECK: toy.print(%alloc) : memref<2x3xf64>
    // CHECK: memref.dealloc %alloc : memref<2x3xf64>
    // CHECK: return
}

// CHECK-LABEL: @convert_constant_to_mid
toy.func @convert_constant_to_mid() {
    %0 = toy.constant {value = dense<[[2.000000e+00, 4.000000e+00, 6.000000e+00], [8.000000e+00, 1.000000e+01, 1.200000e+01]]> : tensor<2x3xf64>} : tensor<2x3xf64>
    toy.print(%0) : tensor<2x3xf64>
    toy.return

    // CHECK: %alloc = memref.alloc() : memref<2x3xf64>
    // CHECK: %c0 = arith.constant 0 : index
    // CHECK: %c1 = arith.constant 1 : index
    // CHECK: %c2 = arith.constant 2 : index
    // CHECK: %cst = arith.constant 2.000000e+00 : f64
    // CHECK: affine.store %cst, %alloc[%c0, %c0] : memref<2x3xf64>
    // CHECK: %cst_0 = arith.constant 4.000000e+00 : f64
    // CHECK: affine.store %cst_0, %alloc[%c0, %c1] : memref<2x3xf64>
    // CHECK: %cst_1 = arith.constant 6.000000e+00 : f64
    // CHECK: affine.store %cst_1, %alloc[%c0, %c2] : memref<2x3xf64>
    // CHECK: %cst_2 = arith.constant 8.000000e+00 : f64
    // CHECK: affine.store %cst_2, %alloc[%c1, %c0] : memref<2x3xf64>
    // CHECK: %cst_3 = arith.constant 1.000000e+01 : f64
    // CHECK: affine.store %cst_3, %alloc[%c1, %c1] : memref<2x3xf64>
    // CHECK: %cst_4 = arith.constant 1.200000e+01 : f64
    // CHECK: affine.store %cst_4, %alloc[%c1, %c2] : memref<2x3xf64>
    // CHECK: toy.print(%alloc) : memref<2x3xf64>
    // CHECK: memref.dealloc %alloc : memref<2x3xf64>
    // CHECK: return
}
