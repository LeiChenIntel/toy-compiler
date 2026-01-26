#ifndef TOY_C_DIALECTS_H
#define TOY_C_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Toy, toy);

#ifdef __cplusplus
}
#endif

#endif // TOY_C_DIALECTS_H
