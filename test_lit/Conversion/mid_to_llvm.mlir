// RUN: toy-opt %s -convert-mid-to-llvm | FileCheck %s

// CHECK-LABEL: @convert_amx_to_llvm
func.func @convert_amx_to_llvm(%arg0: memref<2x4xbf16>, %arg1: memref<2x4xbf16>, %arg2: memref<2x2xf32>) {
    %c0 = arith.constant 0 : index
    %0 = amx.tile_load %arg0[%c0, %c0] : memref<2x4xbf16> into vector<2x4xbf16>
    %1 = amx.tile_load %arg1[%c0, %c0] : memref<2x4xbf16> into vector<2x4xbf16>
    %2 = amx.tile_zero : vector<2x2xf32>
    %3 = amx.tile_mulf %0, %1, %2 : vector<2x4xbf16>, vector<2x4xbf16>, vector<2x2xf32>
    amx.tile_store %arg2[%c0, %c0], %3 : memref<2x2xf32>, vector<2x2xf32>
    return

    // CHECK: [[INPUT_0:%.*]] = "amx.tileloadd64"([[ARG_00:%.*]], [[ARG_01:%.*]], [[ARG_02:%.*]], [[ARG_03:%.*]]) : (i16, i16, !llvm.ptr, i64) -> !llvm.array<2 x vector<4xbf16>>
    // CHECK: [[INPUT_1:%.*]] = "amx.tileloadd64"([[ARG_10:%.*]], [[ARG_11:%.*]], [[ARG_12:%.*]], [[ARG_13:%.*]]) : (i16, i16, !llvm.ptr, i64) -> !llvm.array<2 x vector<4xbf16>>
    // CHECK: [[ZEROS:%.*]] = "amx.tilezero"([[ARG_20:%.*]], [[ARG_21:%.*]]) : (i16, i16) -> !llvm.array<2 x vector<2xf32>>
    // CHECK: [[OUTPUT:%.*]] = "amx.tdpbf16ps"([[ARG_30:%.*]], [[ARG_31:%.*]], [[ARG_32:%.*]], [[ZEROS]], [[INPUT_0]], [[INPUT_1]]) : (i16, i16, i16, !llvm.array<2 x vector<2xf32>>,
    // CHECK-SAME:   !llvm.array<2 x vector<4xbf16>>, !llvm.array<2 x vector<4xbf16>>) -> !llvm.array<2 x vector<2xf32>>
    // CHECK: "amx.tilestored64"([[ARG_40:%.*]], [[ARG_41:%.*]], [[ARG_42:%.*]], [[ARG_43:%.*]], [[OUTPUT]]) : (i16, i16, !llvm.ptr, i64, !llvm.array<2 x vector<2xf32>>) -> ()
}
