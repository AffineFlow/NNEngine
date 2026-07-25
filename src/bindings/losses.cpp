#include <pybind11/pybind11.h>

#include <memory>

#include "core/Loss.hpp"
#include "core/Regularizer.hpp"
#include "losses/MSELoss.hpp"
#include "losses/SoftmaxCrossEntropyLoss.hpp"
#include "regularizers/L2Regularizer.hpp"

namespace py = pybind11;
using namespace mlengine::core;

void bind_losses_and_regs(py::module_& m) {
  py::class_<Loss, std::shared_ptr<Loss>>(m, "Loss").def(
      "forward", &Loss::forward, py::arg("predictions"), py::arg("targets"));

  py::class_<MSELoss, Loss, std::shared_ptr<MSELoss>>(m, "MSELoss")
      .def(py::init<>());

  py::class_<SoftmaxCrossEntropyLoss, Loss,
             std::shared_ptr<SoftmaxCrossEntropyLoss>>(
      m, "SoftmaxCrossEntropyLoss")
      .def(py::init<>());

  py::class_<Regularizer, std::shared_ptr<Regularizer>>(m, "Regularizer");

  py::class_<L2Regularizer, Regularizer, std::shared_ptr<L2Regularizer>>(
      m, "L2Regularizer")
      .def(py::init<float>(), py::arg("l2") = 0.0001f);
}