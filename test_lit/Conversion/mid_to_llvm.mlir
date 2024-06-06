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

    // CHECK: %48 = "amx.tileloadd64"(%41, %40, %47, %42) : (i16, i16, !llvm.ptr, i64) -> !llvm.array<2 x vector<4xbf16>>
    // CHECK: %57 = "amx.tileloadd64"(%50, %49, %56, %51) : (i16, i16, !llvm.ptr, i64) -> !llvm.array<2 x vector<4xbf16>>
    // CHECK: %60 = "amx.tilezero"(%59, %58) : (i16, i16) -> !llvm.array<2 x vector<2xf32>>
    // CHECK: %65 = "amx.tdpbf16ps"(%62, %63, %61, %60, %48, %57) : (i16, i16, i16, !llvm.array<2 x vector<2xf32>>,
    // CHECK-SAME:   !llvm.array<2 x vector<4xbf16>>, !llvm.array<2 x vector<4xbf16>>) -> !llvm.array<2 x vector<2xf32>>
    // CHECK: "amx.tilestored64"(%67, %66, %73, %68, %65) : (i16, i16, !llvm.ptr, i64, !llvm.array<2 x vector<2xf32>>) -> ()
}
