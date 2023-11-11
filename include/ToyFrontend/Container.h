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

#endif // TOY_COMPILER_CONTAINER_H
