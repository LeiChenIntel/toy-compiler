// RUN: toy-opt %s -convert-toy-to-mid=lowering-pattern-mode=1 | FileCheck %s

// CHECK-LABEL: @convert_add_to_mid_vec_one_dim
toy.func @convert_add_to_mid_vec_one_dim(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) {
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

// CHECK-LABEL: @convert_add_to_mid_vec_two_dim
toy.func @convert_add_to_mid_vec_two_dim(%arg0: tensor<2x3xf64>, %arg1: tensor<2x3xf64>) {
    %0 = toy.add(%arg0, %arg1) : tensor<2x3xf64>, tensor<2x3xf64> -> tensor<2x3xf64>
    toy.return

    // CHECK: %alloc = memref.alloc() : memref<2x3xf64>
    // CHECK: %c0 = arith.constant 0 : index
    // %0 = vector.load %arg0[%c0, %c0] : memref<2x3xf64>, vector<2x3xf64>
    // %1 = vector.load %arg1[%c0, %c0] : memref<2x3xf64>, vector<2x3xf64>
    // %2 = arith.addf %0, %1 : vector<2x3xf64>
    // vector.store %2, %alloc[%c0, %c0] : memref<2x3xf64>, vector<2x3xf64>
    // CHECK: memref.dealloc %alloc : memref<2x3xf64>
    // CHECK: return
}

// CHECK-LABEL: @convert_mul_to_mid_vec_one_dim
toy.func @convert_mul_to_mid_vec_one_dim(%arg0: tensor<3xf64>, %arg1: tensor<3xf64>) {
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

// CHECK-LABEL: @convert_mul_to_mid_vec_two_dim
toy.func @convert_mul_to_mid_vec_two_dim(%arg0: tensor<2x3xf64>, %arg1: tensor<2x3xf64>) {
    %0 = toy.mul(%arg0, %arg1) : tensor<2x3xf64>, tensor<2x3xf64> -> tensor<2x3xf64>
    toy.return

    // CHECK: %alloc = memref.alloc() : memref<2x3xf64>
    // CHECK: %c0 = arith.constant 0 : index
    // %0 = vector.load %arg0[%c0, %c0] : memref<2x3xf64>, vector<2x3xf64>
    // %1 = vector.load %arg1[%c0, %c0] : memref<2x3xf64>, vector<2x3xf64>
    // %2 = arith.mulf %0, %1 : vector<2x3xf64>
    // vector.store %2, %alloc[%c0, %c0] : memref<2x3xf64>, vector<2x3xf64>
    // CHECK: memref.dealloc %alloc : memref<2x3xf64>
    // CHECK: return
}
