// RUN: toy-opt %s | FileCheck %s

// CHECK-LABEL: @mlir_toy_scalar
toy.func @mlir_toy_scalar() {
	%0 = toy.constant {value = dense<5.500000e+00> : tensor<f64>} : tensor<f64>
	%1 = toy.reshape(%0) : tensor<f64> -> tensor<2x2xf64>
	toy.print(%1) : tensor<2x2xf64>
	toy.return

    // CHECK: [[VAL_0:%.*]] = toy.constant {value = dense<5.500000e+00> : tensor<f64>} : tensor<f64>
    // CHECK: [[VAL_1:%.*]] = toy.reshape([[VAL_0]]) : tensor<f64> -> tensor<2x2xf64>
    // CHECK: toy.print([[VAL_1]]) : tensor<2x2xf64>
	// CHECK: toy.return
}
