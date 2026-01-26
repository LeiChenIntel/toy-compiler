#include "ToyCAPI/Dialects.h"
#include "Toy/Dialect.h"

#include <mlir/CAPI/Registration.h>

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Toy, toy, mlir::toy::ToyDialect)
