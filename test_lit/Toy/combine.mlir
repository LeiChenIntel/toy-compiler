// RUN: toy-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: @constant_reshape_combine
toy.func @constant_reshape_combine() {
    %0 = toy.constant {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>} : tensor<6xf64>
    %1 = toy.reshape(%0) : tensor<6xf64> -> tensor<2x3xf64>
    toy.print(%1) : tensor<2x3xf64>
    toy.return

    // CHECK: [[VAL_0:%.*]] = toy.constant {value = dense<{{\[\[}}1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : tensor<2x3xf64>
    // CHECK: toy.print([[VAL_0]]) : tensor<2x3xf64>
	// CHECK: toy.return
}
