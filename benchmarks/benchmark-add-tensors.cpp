#include <cassert>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include <immintrin.h>

// Declare function of MLIR generated lib
extern "C" {
void add_tensors(double *src1, double *src2, double *dst);
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

bool checkAccuracy(double *dst, double *src1, double *src2, int n) {
  for (int i = 0; i < n; i++) {
    double ref = src1[i] + src2[i];
    if (ref != dst[i]) {
      printf("Wrong Accuracy in index [%d]\n", i);
      return false;
    }
  }
  return true;
}

void loopNotAligned(const std::vector<int> &elementNum, int iter) {
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

      assert(checkAccuracy(dst, src1, src2, n) == true);
      free(src1);
      free(src2);
      free(dst);
    }

    printf("[ADD LOOP NOT ALIGNED] Elements: %6d", n);
    printf(" Time: %lldns\n", tcnt / iter);
  }
}

void avx2Aligned(const std::vector<int> &elementNum, int iter) {
  for (auto &n : elementNum) {
    long long tcnt = 0;
    for (int t = 0; t < iter; t++) {
      double *src1 = createDoubleAlignedBuffer(n);
      double *src2 = createDoubleAlignedBuffer(n);
      double *dst = createDoubleAlignedBuffer(n);

      const auto start = std::chrono::high_resolution_clock::now();
      int i = 0;
      __m256d c1, c2, c3, c4;
      double *p1 = src1 + i;
      double *p2 = src2 + i;
      double *q1 = dst + i;
      for (; i < n - 8; i += 8) {
        c1 = _mm256_load_pd(p1);
        c2 = _mm256_load_pd(p2);
        c3 = _mm256_load_pd(p1 + 4);
        c4 = _mm256_load_pd(p2 + 4);
        c1 = _mm256_add_pd(c1, c2);
        c3 = _mm256_add_pd(c3, c4);
        _mm256_store_pd(q1, c1);
        _mm256_store_pd(q1 + 4, c3);
        p1 += 8;
        p2 += 8;
        q1 += 8;
      }
      for (; i < n; i++) {
        dst[i] = src1[i] + src2[i];
      }
      const auto end = std::chrono::high_resolution_clock::now();
      const auto last =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
      tcnt += last.count();

      assert(checkAccuracy(dst, src1, src2, n) == true);
      _mm_free(src1);
      _mm_free(src2);
      _mm_free(dst);
    }
    printf("[ADD AVX2 ALIGNED] Elements: %6d", n);
    printf(" Time: %lldns\n", tcnt / iter);
  }
}

void avx2MLIRAligned(const std::vector<int> &elementNum, int iter) {
  for (auto &n : elementNum) {
    long long tcnt = 0;
    for (int t = 0; t < iter; t++) {
      double *src1 = createDoubleAlignedBuffer(n);
      double *src2 = createDoubleAlignedBuffer(n);
      double *dst = createDoubleAlignedBuffer(n);

      const auto start = std::chrono::high_resolution_clock::now();
      add_tensors(src1, src2, dst);
      const auto end = std::chrono::high_resolution_clock::now();
      const auto last =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
      tcnt += last.count();

      assert(checkAccuracy(dst, src1, src2, n) == true);
      _mm_free(src1);
      _mm_free(src2);
      _mm_free(dst);
    }
    printf("[MLIR ADD AVX2 ALIGNED] Elements: %6d", n);
    printf(" Time: %lldns\n", tcnt / iter);
  }
}

int main() {
  const std::vector<int> elementNum{8192, 16384, 32768, 65536, 131072, 262144};
  const int iter = 100;
  std::vector<long long> loopResults;
  std::vector<long long> avx2Results;

  // Loop and not aligned
  loopNotAligned(elementNum, iter);

  // AVX2 and aligned
  avx2Aligned(elementNum, iter);

  // MLIR AVX2 and aligned
  const std::vector<int> elementNum2{8192};
  avx2MLIRAligned(elementNum2, iter);

  return 0;
}
