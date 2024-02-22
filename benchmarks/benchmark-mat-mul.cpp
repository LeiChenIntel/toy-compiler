#include <iostream>

#include <cblas.h>

int main() {
  int i = 0;
  double A[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  double B[6] = {7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
  double C[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, 3, 3, 2, 1, A, 3, B, 3,
              0, C, 3);

  printf("%lf ", A[0]);
  printf("%lf ", B[0]);

  for (i = 0; i < 9; i++)
    printf("%lf ", C[i]);
  printf("\n");
  return 0;
}
