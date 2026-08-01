#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

#include "core/Optimizer.hpp"
#include "core/Scheduler.hpp"
#include "optimizers/Adam.hpp"
#include "optimizers/AdamW.hpp"
#include "optimizers/SGD.hpp"

namespace py = pybind11;
using namespace affineflow::core;

void bind_optimizers(py::module_& m) {
  py::class_<Optimizer, std::shared_ptr<Optimizer>>(
      m, "Optimizer", "Base class for parameter update rules.")
      .def("set_parameters", &Optimizer::set_parameters, py::arg("params"),
           "Bind the trainable tensors to the optimizer.")
      .def("step", &Optimizer::step, "Execute a single optimization step.")
      .def("zero_grad", &Optimizer::zero_grad,
           "Clear the gradients of all optimized parameters.");

  py::class_<SGD, Optimizer, std::shared_ptr<SGD>>(
      m, "SGD", "Plain stochastic gradient descent optimizer.")
      .def(py::init<float>(), py::arg("learning_rate") = 0.01f,
           "Initialize SGD with learning rate.");

  py::class_<Adam, Optimizer, std::shared_ptr<Adam>>(
      m, "Adam", "Adaptive Moment Estimation optimizer.")
      .def(py::init<float>(), py::arg("learning_rate") = 0.001f,
           "Initialize Adam with learning rate.");

  py::class_<AdamW, Optimizer, std::shared_ptr<AdamW>>(
      m, "AdamW", "Adam optimizer with decoupled weight decay.")
      .def(py::init<float, float>(), py::arg("learning_rate") = 0.001f,
           py::arg("weight_decay") = 0.01f,
           "Initialize AdamW with learning rate and decay penalty.");

  py::class_<Scheduler, std::shared_ptr<Scheduler>>(
      m, "Scheduler", "Base class for learning rate scheduling.");

  py::class_<StepLR, Scheduler, std::shared_ptr<StepLR>>(
      m, "StepLR", "Decays the learning rate by gamma every step_size epochs.")
      .def(py::init<std::shared_ptr<Optimizer>, int, float>(),
           py::arg("optimizer"), py::arg("step_size"), py::arg("gamma") = 0.1f,
           "Initialize the StepLR scheduler.")
      .def("step", &StepLR::step, "Update the optimizer's learning rate.");
}