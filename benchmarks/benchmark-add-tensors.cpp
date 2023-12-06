#include "ToyFrontend/Container.h"

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include <immintrin.h>

// Declare function of MLIR generated lib
extern "C" {
void add_tensors(MemRef *src1, MemRef *src2, MemRef *dst);
}

double *createDoubleBuffer(const int n) {
  double *ptr = (double *)malloc(n * sizeof(double));
  std::default_random_engine gen;
  std::uniform_real_distribution<double> dis(-10.0, 10.0);
  for (int i = 0; i < n; i++) {
    ptr[i] = dis(gen);
  }
  return ptr;
}

double *createDoubleAlignedBuffer(const int n) {
  double *ptr = (double *)_mm_malloc(n * sizeof(double), 32);
  std::default_random_engine gen;
  std::uniform_real_distribution<double> dis(-10.0, 10.0);
  for (int i = 0; i < n; i++) {
    ptr[i] = dis(gen);
  }
  return ptr;
}

int main() {
  const std::vector<int> elementNum{8192, 16384, 32768, 65536, 131072, 262144};
  const int iter = 100;
  std::vector<long long> loopResults;
  std::vector<long long> avx2Results;

  // Loop and not aligned
  for (auto &n : elementNum) {
    long long tcnt = 0;
    for (int t = 0; t < iter; t++) {
      double *src1 = createDoubleBuffer(n);
      double *src2 = createDoubleBuffer(n);
      double *dst = createDoubleBuffer(n);

      const auto start = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < n; i++) {
        dst[i] = src1[i] + src2[i];
      }
      const auto end = std::chrono::high_resolution_clock::now();
      const auto last =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
      tcnt += last.count();

      free(src1);
      free(src2);
      free(dst);
    }

    printf("[ADD LOOP NOT ALIGNED] Elements: %6d", n);
    printf(" Time: %lldns\n", tcnt / iter);
  }

  // AVX2 and aligned
  for (auto &n : elementNum) {
    long long tcnt = 0;
    for (int t = 0; t < iter; t++) {
      double *src1 = createDoubleAlignedBuffer(n);
      double *src2 = createDoubleAlignedBuffer(n);
      double *dst = createDoubleAlignedBuffer(n);

      const auto start = std::chrono::high_resolution_clock::now();
      int i = 0;
      __m256d c1;
      __m256d c2;
      __m256d d;
      for (; i < n - 4; i += 4) {
        c1 = _mm256_load_pd(src1 + i);
        c2 = _mm256_load_pd(src2 + i);
        d = _mm256_add_pd(c1, c2);
        _mm256_store_pd(dst + i, d);
      }
      for (; i < n; i++) {
        dst[i] = src1[i] + src2[i];
      }
      const auto end = std::chrono::high_resolution_clock::now();
      const auto last =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
      tcnt += last.count();

      _mm_free(src1);
      _mm_free(src2);
      _mm_free(dst);
    }
    printf("[ADD AVX2 ALIGNED] Elements: %6d", n);
    printf(" Time: %lldns\n", tcnt / iter);
  }

  // MLIR AVX2 and aligned
  const std::vector<int> elementNum2{8192};
  for (auto &n : elementNum2) {
    long long tcnt = 0;
    for (int t = 0; t < iter; t++) {
      double *src1 = createDoubleAlignedBuffer(n);
      double *src2 = createDoubleAlignedBuffer(n);
      double *dst = createDoubleAlignedBuffer(n);

      MemRef input1{src1, src1, 0, n, 0};
      MemRef input2{src2, src2, 0, n, 0};
      MemRef output{dst, dst, 0, n, 0};

      const auto start = std::chrono::high_resolution_clock::now();
      add_tensors(&input1, &input2, &output);
      const auto end = std::chrono::high_resolution_clock::now();
      const auto last =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
      tcnt += last.count();

      printf("%lf\n", src1[0]);
      printf("%lf\n", src2[0]);
      printf("%lf\n", dst[0]);
      printf("%lf\n", src1[1]);
      printf("%lf\n", src2[1]);
      printf("%lf\n", dst[1]);

      _mm_free(src1);
      _mm_free(src2);
      _mm_free(dst);
    }
    printf("[MLIR ADD AVX2 ALIGNED] Elements: %6d", n);
    printf(" Time: %lldns\n", tcnt / iter);
  }

  return 0;
}
