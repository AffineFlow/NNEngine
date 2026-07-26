#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_core_utils(py::module_& m);
void bind_losses_and_regs(py::module_& m);
void bind_optimizers(py::module_& m);
void bind_layers(py::module_& m);

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

PYBIND11_MODULE(BACKEND_NAME, m) {
  m.doc() = "C++ Core Backend Engine for NNEngine";
  bind_core_utils(m);
  bind_losses_and_regs(m);
  bind_optimizers(m);
  bind_layers(m);
}