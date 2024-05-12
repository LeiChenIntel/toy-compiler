#include <cassert>
#include <chrono>
#include <iostream>
#include <random>

#include <cblas.h>
#include <immintrin.h>

double *createDoubleMatrixBuffer(const int row, const int col) {
  double *ptr = (double *)_mm_malloc(row * col * sizeof(double), 32);
  std::default_random_engine gen;
  std::uniform_real_distribution<double> dis(-10.0, 10.0);
  for (int i = 0; i < row * col; i++) {
    ptr[i] = dis(gen);
  }
  return ptr;
}

bfloat16 *createBF16MatrixBuffer(const int row, const int col) {
  auto *ptr = (bfloat16 *)_mm_malloc(row * col * sizeof(bfloat16), 32);
  // bfloat16 refered as unsigned short int in OpenBLAS
  std::default_random_engine gen;
  std::uniform_int_distribution<int> dis(0, 10);
  for (int i = 0; i < row * col; i++) {
    ptr[i] = (bfloat16)dis(gen);
  }
  return ptr;
}

void squareMatmul(double *dst, const double *src1, const double *src2, int n) {
  for (int i = 0; i < n * n; i++) {
    const int r = i / n;
    const int c = i % n;
    // k represents cols of A and rows of B
    double t = 0.0f;
    for (int k = 0; k < n; k++) {
      t = t + src1[c + n * k] * src2[r + n * k]; // rows A * k, cols B * k
    }
    dst[i] = t;
  }
}

bool checkAccuracy(double *dst, double *src1, double *src2, int n) {
  double *ref = createDoubleMatrixBuffer(n, n);
  squareMatmul(ref, src1, src2, n);
  for (int i = 0; i < n * n; i++) {
    if (std::abs(ref[i] - dst[i]) > 0.001) {
      printf("Wrong Accuracy in index [%d]\n", i);
      return false;
    }
  }
  return true;
}

void simpleMatmul(const std::vector<int> &squareMatDim, int iter) {
  for (auto &n : squareMatDim) {
    long long tcnt = 0;
    for (int t = 0; t < iter; t++) {
      double *src1 = createDoubleMatrixBuffer(n, n);
      double *src2 = createDoubleMatrixBuffer(n, n);
      double *dst = createDoubleMatrixBuffer(n, n);

      const auto start = std::chrono::high_resolution_clock::now();
      squareMatmul(dst, src1, src2, n);
      const auto end = std::chrono::high_resolution_clock::now();
      const auto last =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
      tcnt += last.count();

      assert(checkAccuracy(dst, src1, src2, n) == true);
      free(src1);
      free(src2);
      free(dst);
    }

    printf("[Simple MATMUL] SquareDim: %6d", n);
    printf(" Time: %lldns\n", tcnt / iter);
  }
}

void openBLASMatmul(const std::vector<int> &squareMatDim, int iter) {
  for (auto &n : squareMatDim) {
    long long tcnt = 0;
    for (int t = 0; t < iter; t++) {
      double *src1 = createDoubleMatrixBuffer(n, n);
      double *src2 = createDoubleMatrixBuffer(n, n);
      double *dst = createDoubleMatrixBuffer(n, n);
      const auto start = std::chrono::high_resolution_clock::now();
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n, n, n, 1, src1, n,
                  src2, n, 0, dst, n);
      const auto end = std::chrono::high_resolution_clock::now();
      const auto last =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
      tcnt += last.count();

      assert(checkAccuracy(dst, src1, src2, n) == true);
      free(src1);
      free(src2);
      free(dst);
    }

    printf("[OpenBLAS MATMUL] SquareDim: %6d", n);
    printf(" Time: %lldns\n", tcnt / iter);
  }
}

void clean() {
  const int size = 100 * 1024 * 1024; // Allocate 100M. Set much larger than L2
  char *c = (char *)malloc(size);
  for (int i = 0; i < 0xffff; i++)
    for (int j = 0; j < size; j++)
      c[j] = i * j;
}

int main() {
  const std::vector<int> squareMatDim{8, 16, 32, 64, 128, 256};
  const int iter = 100;

  // Manually matrix multiplication one element by one element
  // clean();
  simpleMatmul(squareMatDim, iter);

  // OpenBLAS matrix multiplication
  // clean();
  openBLASMatmul(squareMatDim, iter);

  return 0;
}
