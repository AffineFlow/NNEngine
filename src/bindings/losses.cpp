#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

#include "core/Loss.hpp"
#include "core/Regularizer.hpp"
#include "losses/MSELoss.hpp"
#include "losses/SoftmaxCrossEntropyLoss.hpp"
#include "regularizers/L2Regularizer.hpp"

namespace py = pybind11;
using namespace affineengine::core;

void bind_losses_and_regs(py::module_& m) {
  py::class_<Loss, std::shared_ptr<Loss>>(
      m, "Loss", "Objective function that produces a scalar training signal.")
      .def("forward", &Loss::forward, py::arg("predictions"),
           py::arg("targets"),
           "Bind tensors and compute the forward loss scalar.")
      .def("backward", &Loss::backward,
           "Explicitly seed the gradient into the prediction tensor.");

  py::class_<MSELoss, Loss, std::shared_ptr<MSELoss>>(
      m, "MSELoss", "Mean-squared-error objective for regression.")
      .def(py::init<>(), "Initialize MSELoss.");

  py::class_<SoftmaxCrossEntropyLoss, Loss,
             std::shared_ptr<SoftmaxCrossEntropyLoss>>(
      m, "SoftmaxCrossEntropyLoss",
      "Numerically stable softmax cross-entropy for classification.")
      .def(py::init<>(), "Initialize SoftmaxCrossEntropyLoss.");

  py::class_<Regularizer, std::shared_ptr<Regularizer>>(
      m, "Regularizer",
      "Penalty term applied to trainable parameters during optimization.")
      .def("apply", &Regularizer::apply, py::arg("parameters"),
           "Accumulate a regularization penalty and any gradient adjustment.");

  py::class_<L2Regularizer, Regularizer, std::shared_ptr<L2Regularizer>>(
      m, "L2Regularizer", "L2 weight decay regularizer.")
      .def(py::init<float>(), py::arg("l2") = 0.0001f,
           "Construct with a scaling penalty coefficient.");
}