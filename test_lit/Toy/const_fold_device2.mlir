// RUN: toy-opt %s -canonicalize --toy-platform=device2 | FileCheck %s
// Const fold is not included in the device 2 patterns, so IR will remain the same

// CHECK-LABEL: @add_constant_fold
toy.func @add_constant_fold() {
    %0 = toy.constant {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>} : tensor<6xf64>
    %1 = toy.reshape(%0) : tensor<6xf64> -> tensor<2x3xf64>
    %2 = toy.constant {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>} : tensor<6xf64>
    %3 = toy.reshape(%2) : tensor<6xf64> -> tensor<2x3xf64>
    %4 = toy.add(%1, %3) : tensor<2x3xf64>, tensor<2x3xf64> -> tensor<2x3xf64>
    toy.print(%4) : tensor<2x3xf64>
    toy.return

    // CHECK: [[VAL_0:%.*]] = toy.constant {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>} : tensor<6xf64>
    // CHECK: [[VAL_1:%.*]] = toy.reshape([[VAL_0]]) : tensor<6xf64> -> tensor<2x3xf64>
    // CHECK: [[VAL_2:%.*]] = toy.constant {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>} : tensor<6xf64>
    // CHECK: [[VAL_3:%.*]] = toy.reshape([[VAL_2]]) : tensor<6xf64> -> tensor<2x3xf64>
    // CHECK: [[VAL_4:%.*]] = toy.add([[VAL_1]], [[VAL_3]]) : tensor<2x3xf64>, tensor<2x3xf64> -> tensor<2x3xf64>
    // CHECK: toy.print([[VAL_4]]) : tensor<2x3xf64>
    // CHECK: toy.return
}

// CHECK-LABEL: @mul_constant_fold
toy.func @mul_constant_fold() {
    %0 = toy.constant {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>} : tensor<6xf64>
    %1 = toy.reshape(%0) : tensor<6xf64> -> tensor<2x3xf64>
    %2 = toy.constant {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>} : tensor<6xf64>
    %3 = toy.reshape(%2) : tensor<6xf64> -> tensor<2x3xf64>
    %4 = toy.mul(%1, %3) : tensor<2x3xf64>, tensor<2x3xf64> -> tensor<2x3xf64>
    toy.print(%4) : tensor<2x3xf64>
    toy.return

    // CHECK: [[VAL_0:%.*]] = toy.constant {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>} : tensor<6xf64>
    // CHECK: [[VAL_1:%.*]] = toy.reshape([[VAL_0]]) : tensor<6xf64> -> tensor<2x3xf64>
    // CHECK: [[VAL_2:%.*]] = toy.constant {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>} : tensor<6xf64>
    // CHECK: [[VAL_3:%.*]] = toy.reshape([[VAL_2]]) : tensor<6xf64> -> tensor<2x3xf64>
    // CHECK: [[VAL_4:%.*]] = toy.mul([[VAL_1]], [[VAL_3]]) : tensor<2x3xf64>, tensor<2x3xf64> -> tensor<2x3xf64>
    // CHECK: toy.print([[VAL_4]]) : tensor<2x3xf64>
    // CHECK: toy.return
}
