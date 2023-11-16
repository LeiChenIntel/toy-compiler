#include "Toy/Dialect.h"
#include <mlir/Bindings/Python/PybindAdaptors.h>

namespace py = pybind11;
using namespace mlir::python::adaptors;

PYBIND11_MODULE(_toyDialects, m) {
  auto toyModule = m.def_submodule("toy");

  toyModule.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__to__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);
}
