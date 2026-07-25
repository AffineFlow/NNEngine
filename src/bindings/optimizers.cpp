#include <pybind11/pybind11.h>

#include <memory>

#include "core/Optimizer.hpp"
#include "core/Scheduler.hpp"
#include "optimizers/Adam.hpp"
#include "optimizers/AdamW.hpp"
#include "optimizers/SGD.hpp"

namespace py = pybind11;
using namespace mlengine::core;

void bind_optimizers(py::module_& m) {
  py::class_<Optimizer, std::shared_ptr<Optimizer>>(m, "Optimizer");

  py::class_<SGD, Optimizer, std::shared_ptr<SGD>>(m, "SGD").def(
      py::init<float>(), py::arg("learning_rate") = 0.01f);

  py::class_<Adam, Optimizer, std::shared_ptr<Adam>>(m, "Adam").def(
      py::init<float>(), py::arg("learning_rate") = 0.001f);

  py::class_<AdamW, Optimizer, std::shared_ptr<AdamW>>(m, "AdamW")
      .def(py::init<float, float>(), py::arg("learning_rate") = 0.001f,
           py::arg("weight_decay") = 0.01f);

  py::class_<Scheduler, std::shared_ptr<Scheduler>>(m, "Scheduler");

  py::class_<StepLR, Scheduler, std::shared_ptr<StepLR>>(m, "StepLR")
      .def(py::init<std::shared_ptr<Optimizer>, int, float>(),
           py::arg("optimizer"), py::arg("step_size"), py::arg("gamma") = 0.1f)
      .def("step", &StepLR::step);
}