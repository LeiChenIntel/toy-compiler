#ifndef TOY_COMPILER_CONTAINER_H
#define TOY_COMPILER_CONTAINER_H

#include <memory>

struct MemRef {
  double *allocated = nullptr;
  double *aligned = nullptr;
  intptr_t offset = 0;
  intptr_t size;
  intptr_t stride;
};

struct MemRef2d {
  double *allocated = nullptr;
  double *aligned = nullptr;
  intptr_t offset = 0;
  intptr_t size[2];
  intptr_t stride[2];
};

#endif // TOY_COMPILER_CONTAINER_H
