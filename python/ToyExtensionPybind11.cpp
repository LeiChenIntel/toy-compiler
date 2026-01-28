#include "ToyCAPI/Dialects.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

using namespace mlir::python::adaptors;

PYBIND11_MODULE(_toyDialectsPybind11, m) {
  //===--------------------------------------------------------------------===//
  // toy dialect
  //===--------------------------------------------------------------------===//
  auto standaloneM = m.def_submodule("toy");

  standaloneM.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__toy__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);
}
