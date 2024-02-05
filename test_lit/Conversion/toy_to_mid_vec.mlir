// RUN: toy-opt %s -convert-toy-to-mid=lowering-pattern-mode=1 | FileCheck %s

// CHECK-LABEL: @convert_add_to_mid_vec_3_dim
toy.func @convert_add_to_mid_vec_3_dim(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) {
    %0 = toy.add(%arg0, %arg1) : tensor<3xf64>, tensor<3xf64> -> tensor<3xf64>
    toy.return

    // CHECK: %alloc = memref.alloc() : memref<3xf64>
    // CHECK: %c0 = arith.constant 0 : index
    // CHECK: %0 = vector.load %arg0[%c0] : memref<3xf64>, vector<3xf64>
    // CHECK: %1 = vector.load %arg1[%c0] : memref<3xf64>, vector<3xf64>
    // CHECK: %2 = arith.addf %0, %1 : vector<3xf64>
    // CHECK: vector.store %2, %alloc[%c0] : memref<3xf64>, vector<3xf64>
    // CHECK: memref.dealloc %alloc : memref<3xf64>
    // CHECK: return
}

// CHECK-LABEL: @convert_add_to_mid_vec_17_dim
toy.func @convert_add_to_mid_vec_17_dim(%arg0: tensor<17xf64>, %arg1: tensor<17xf64>) {
    %0 = toy.add(%arg0, %arg1) : tensor<17xf64>, tensor<17xf64> -> tensor<17xf64>
    toy.return

    // CHECK: %alloc = memref.alloc() : memref<17xf64>
    // CHECK: affine.for %arg2 = 0 to 16 step 16 {
    // CHECK:  %3 = vector.load %arg0[%arg2] : memref<17xf64>, vector<16xf64>
    // CHECK:  %4 = vector.load %arg1[%arg2] : memref<17xf64>, vector<16xf64>
    // CHECK:  %5 = arith.addf %3, %4 : vector<16xf64>
    // CHECK:  vector.store %5, %alloc[%arg2] : memref<17xf64>, vector<16xf64>
    // CHECK: }
    // CHECK: %c16 = arith.constant 16 : index
    // CHECK: %0 = vector.load %arg0[%c16] : memref<17xf64>, vector<1xf64>
    // CHECK: %1 = vector.load %arg1[%c16] : memref<17xf64>, vector<1xf64>
    // CHECK: %2 = arith.addf %0, %1 : vector<1xf64>
    // CHECK: vector.store %2, %alloc[%c16] : memref<17xf64>, vector<1xf64>
    // CHECK: memref.dealloc %alloc : memref<17xf64>
    // CHECK: return
}

// CHECK-LABEL: @convert_add_to_mid_vec_2x3_dim
toy.func @convert_add_to_mid_vec_2x3_dim(%arg0: tensor<2x3xf64>, %arg1: tensor<2x3xf64>) {
    %0 = toy.add(%arg0, %arg1) : tensor<2x3xf64>, tensor<2x3xf64> -> tensor<2x3xf64>
    toy.return

    // CHECK: %alloc = memref.alloc() : memref<2x3xf64>
    // CHECK: %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [6], strides: [1] : memref<2x3xf64> to memref<6xf64>
    // CHECK: %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [6], strides: [1] : memref<2x3xf64> to memref<6xf64>
    // CHECK: %reinterpret_cast_1 = memref.reinterpret_cast %alloc to offset: [0], sizes: [6], strides: [1] : memref<2x3xf64> to memref<6xf64>
    // CHECK: %c0 = arith.constant 0 : index
    // CHECK: %0 = vector.load %reinterpret_cast[%c0] : memref<6xf64>, vector<6xf64>
    // CHECK: %1 = vector.load %reinterpret_cast_0[%c0] : memref<6xf64>, vector<6xf64>
    // CHECK: %2 = arith.addf %0, %1 : vector<6xf64>
    // CHECK: vector.store %2, %reinterpret_cast_1[%c0] : memref<6xf64>, vector<6xf64>
    // CHECK: memref.dealloc %alloc : memref<2x3xf64>
    // CHECK: return
}

// CHECK-LABEL: @convert_add_to_mid_vec_2x9_dim
toy.func @convert_add_to_mid_vec_2x9_dim(%arg0: tensor<2x9xf64>, %arg1: tensor<2x9xf64>) {
    %0 = toy.add(%arg0, %arg1) : tensor<2x9xf64>, tensor<2x9xf64> -> tensor<2x9xf64>
    toy.return

    // CHECK: %alloc = memref.alloc() : memref<2x9xf64>
    // CHECK: %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [18], strides: [1] : memref<2x9xf64> to memref<18xf64>
    // CHECK: %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [18], strides: [1] : memref<2x9xf64> to memref<18xf64>
    // CHECK: %reinterpret_cast_1 = memref.reinterpret_cast %alloc to offset: [0], sizes: [18], strides: [1] : memref<2x9xf64> to memref<18xf64>
    // CHECK: affine.for %arg2 = 0 to 16 step 16 {
    // CHECK:  %3 = vector.load %reinterpret_cast[%arg2] : memref<18xf64>, vector<16xf64>
    // CHECK:  %4 = vector.load %reinterpret_cast_0[%arg2] : memref<18xf64>, vector<16xf64>
    // CHECK:  %5 = arith.addf %3, %4 : vector<16xf64>
    // CHECK:  vector.store %5, %reinterpret_cast_1[%arg2] : memref<18xf64>, vector<16xf64>
    // CHECK: }
    // CHECK: %c16 = arith.constant 16 : index
    // CHECK: %0 = vector.load %reinterpret_cast[%c16] : memref<18xf64>, vector<2xf64>
    // CHECK: %1 = vector.load %reinterpret_cast_0[%c16] : memref<18xf64>, vector<2xf64>
    // CHECK: %2 = arith.addf %0, %1 : vector<2xf64>
    // CHECK: vector.store %2, %reinterpret_cast_1[%c16] : memref<18xf64>, vector<2xf64>
    // CHECK: memref.dealloc %alloc : memref<2x9xf64>
    // CHECK: return
}

// CHECK-LABEL: @convert_mul_to_mid_vec_3_dim
toy.func @convert_mul_to_mid_vec_3_dim(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) {
    %0 = toy.mul(%arg0, %arg1) : tensor<3xf64>, tensor<3xf64> -> tensor<3xf64>
    toy.return

    // CHECK: %alloc = memref.alloc() : memref<3xf64>
    // CHECK: %c0 = arith.constant 0 : index
    // CHECK: %0 = vector.load %arg0[%c0] : memref<3xf64>, vector<3xf64>
    // CHECK: %1 = vector.load %arg1[%c0] : memref<3xf64>, vector<3xf64>
    // CHECK: %2 = arith.mulf %0, %1 : vector<3xf64>
    // CHECK: vector.store %2, %alloc[%c0] : memref<3xf64>, vector<3xf64>
    // CHECK: memref.dealloc %alloc : memref<3xf64>
    // CHECK: return
}

// CHECK-LABEL: @convert_mul_to_mid_vec_17_dim
toy.func @convert_mul_to_mid_vec_17_dim(%arg0: tensor<17xf64>, %arg1: tensor<17xf64>) {
    %0 = toy.mul(%arg0, %arg1) : tensor<17xf64>, tensor<17xf64> -> tensor<17xf64>
    toy.return

    // CHECK: %alloc = memref.alloc() : memref<17xf64>
    // CHECK: affine.for %arg2 = 0 to 16 step 16 {
    // CHECK:  %3 = vector.load %arg0[%arg2] : memref<17xf64>, vector<16xf64>
    // CHECK:  %4 = vector.load %arg1[%arg2] : memref<17xf64>, vector<16xf64>
    // CHECK:  %5 = arith.mulf %3, %4 : vector<16xf64>
    // CHECK:  vector.store %5, %alloc[%arg2] : memref<17xf64>, vector<16xf64>
    // CHECK: }
    // CHECK: %c16 = arith.constant 16 : index
    // CHECK: %0 = vector.load %arg0[%c16] : memref<17xf64>, vector<1xf64>
    // CHECK: %1 = vector.load %arg1[%c16] : memref<17xf64>, vector<1xf64>
    // CHECK: %2 = arith.mulf %0, %1 : vector<1xf64>
    // CHECK: vector.store %2, %alloc[%c16] : memref<17xf64>, vector<1xf64>
    // CHECK: memref.dealloc %alloc : memref<17xf64>
    // CHECK: return
}

// CHECK-LABEL: @convert_mul_to_mid_vec_2x3_dim
toy.func @convert_mul_to_mid_vec_2x3_dim(%arg0: tensor<2x3xf64>, %arg1: tensor<2x3xf64>) {
    %0 = toy.mul(%arg0, %arg1) : tensor<2x3xf64>, tensor<2x3xf64> -> tensor<2x3xf64>
    toy.return

    // CHECK: %alloc = memref.alloc() : memref<2x3xf64>
    // CHECK: %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [6], strides: [1] : memref<2x3xf64> to memref<6xf64>
    // CHECK: %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [6], strides: [1] : memref<2x3xf64> to memref<6xf64>
    // CHECK: %reinterpret_cast_1 = memref.reinterpret_cast %alloc to offset: [0], sizes: [6], strides: [1] : memref<2x3xf64> to memref<6xf64>
    // CHECK: %c0 = arith.constant 0 : index
    // CHECK: %0 = vector.load %reinterpret_cast[%c0] : memref<6xf64>, vector<6xf64>
    // CHECK: %1 = vector.load %reinterpret_cast_0[%c0] : memref<6xf64>, vector<6xf64>
    // CHECK: %2 = arith.mulf %0, %1 : vector<6xf64>
    // CHECK: vector.store %2, %reinterpret_cast_1[%c0] : memref<6xf64>, vector<6xf64>
    // CHECK: memref.dealloc %alloc : memref<2x3xf64>
    // CHECK: return
}

// CHECK-LABEL: @convert_mul_to_mid_vec_4x5_dim
toy.func @convert_mul_to_mid_vec_4x5_dim(%arg0: tensor<4x5xf64>, %arg1: tensor<4x5xf64>) {
    %0 = toy.mul(%arg0, %arg1) : tensor<4x5xf64>, tensor<4x5xf64> -> tensor<4x5xf64>
    toy.return

    // CHECK: %alloc = memref.alloc() : memref<4x5xf64>
    // CHECK: %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [20], strides: [1] : memref<4x5xf64> to memref<20xf64>
    // CHECK: %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [20], strides: [1] : memref<4x5xf64> to memref<20xf64>
    // CHECK: %reinterpret_cast_1 = memref.reinterpret_cast %alloc to offset: [0], sizes: [20], strides: [1] : memref<4x5xf64> to memref<20xf64>
    // CHECK: affine.for %arg2 = 0 to 16 step 16 {
    // CHECK:  %3 = vector.load %reinterpret_cast[%arg2] : memref<20xf64>, vector<16xf64>
    // CHECK:  %4 = vector.load %reinterpret_cast_0[%arg2] : memref<20xf64>, vector<16xf64>
    // CHECK:  %5 = arith.mulf %3, %4 : vector<16xf64>
    // CHECK:  vector.store %5, %reinterpret_cast_1[%arg2] : memref<20xf64>, vector<16xf64>
    // CHECK: }
    // CHECK: %c16 = arith.constant 16 : index
    // CHECK: %0 = vector.load %reinterpret_cast[%c16] : memref<20xf64>, vector<4xf64>
    // CHECK: %1 = vector.load %reinterpret_cast_0[%c16] : memref<20xf64>, vector<4xf64>
    // CHECK: %2 = arith.mulf %0, %1 : vector<4xf64>
    // CHECK: vector.store %2, %reinterpret_cast_1[%c16] : memref<20xf64>, vector<4xf64>
    // CHECK: memref.dealloc %alloc : memref<4x5xf64>
    // CHECK: return
}
