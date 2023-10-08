// RUN: toy-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: @add_constant_fold
toy.func @add_constant_fold() {
    %0 = toy.constant {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>} : tensor<6xf64>
    %1 = toy.reshape(%0) : tensor<6xf64> -> tensor<2x3xf64>
    %2 = toy.constant {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>} : tensor<6xf64>
    %3 = toy.reshape(%2) : tensor<6xf64> -> tensor<2x3xf64>
    %4 = toy.add(%1, %3) : tensor<2x3xf64>, tensor<2x3xf64> -> tensor<2x3xf64>
    toy.print(%4) : tensor<2x3xf64>
    toy.return

    // CHECK: [[VAL_0:%.*]] = toy.constant {value = dense<{{\[\[}}2.000000e+00, 4.000000e+00, 6.000000e+00], [8.000000e+00, 1.000000e+01, 1.200000e+01]]> : tensor<2x3xf64>} : tensor<2x3xf64>
    // CHECK: toy.print([[VAL_0]]) : tensor<2x3xf64>
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

    // CHECK: [[VAL_0:%.*]] = toy.constant {value = dense<{{\[\[}}1.000000e+00, 4.000000e+00, 9.000000e+00], [1.600000e+01, 2.500000e+01, 3.600000e+01]]> : tensor<2x3xf64>} : tensor<2x3xf64>
    // CHECK: toy.print([[VAL_0]]) : tensor<2x3xf64>
    // CHECK: toy.return
}
