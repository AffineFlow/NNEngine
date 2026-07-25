#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

#include "core/JITGraph.hpp"
#include "core/Layer.hpp"
#include "core/Module.hpp"
#include "layers/BatchNorm1dLayer.hpp"
#include "layers/Conv2dLayer.hpp"
#include "layers/DenseLayer.hpp"
#include "layers/DropoutLayer.hpp"
#include "layers/LeakyReLULayer.hpp"
#include "layers/ReLULayer.hpp"

namespace py = pybind11;
using namespace mlengine::core;
using namespace mlengine::layers;

class PyModule : public Module {
 public:
  using Module::Module;
  mlengine::autograd::Tensor* forward(
      mlengine::autograd::Tensor* input) override {
    PYBIND11_OVERRIDE_PURE(mlengine::autograd::Tensor*, Module, forward, input);
  }
};

void bind_layers(py::module_& m) {
  py::class_<Layer, std::shared_ptr<Layer>>(m, "Layer")
      .def("parameters", &Layer::parameters, py::return_value_policy::reference)
      .def("forward", &Layer::forward, py::return_value_policy::reference)
      .def("__call__", &Layer::forward, py::return_value_policy::reference)
      .def("train", &Layer::train, py::arg("mode") = true)
      .def("eval", &Layer::eval);

  py::class_<Module, Layer, PyModule, std::shared_ptr<Module>>(m, "Module")
      .def(py::init<>())
      .def("forward", &Module::forward, py::return_value_policy::reference)
      .def("__call__", &Module::forward, py::return_value_policy::reference)
      .def("predict", &Module::predict, py::arg("X"),
           py::call_guard<py::gil_scoped_release>())
      .def("parameters", &Module::parameters,
           py::return_value_policy::reference)
      .def("named_parameters", &Module::named_parameters,
           py::return_value_policy::reference_internal)
      .def("save_weights", &Module::save_weights, py::arg("filepath"),
           py::call_guard<py::gil_scoped_release>())
      .def("load_weights", &Module::load_weights, py::arg("filepath"),
           py::call_guard<py::gil_scoped_release>())
      .def("register_module", &Module::register_module, py::arg("name"),
           py::arg("layer"), py::return_value_policy::reference);

  py::class_<DenseLayer, Layer, std::shared_ptr<DenseLayer>>(m, "DenseLayer")
      .def(py::init<int, int>())
      .def("forward", &DenseLayer::forward, py::return_value_policy::reference)
      .def("__call__", &DenseLayer::forward,
           py::return_value_policy::reference);

  py::class_<ReLULayer, Layer, std::shared_ptr<ReLULayer>>(m, "ReLULayer")
      .def(py::init<>())
      .def("forward", &ReLULayer::forward, py::return_value_policy::reference)
      .def("__call__", &ReLULayer::forward, py::return_value_policy::reference);

  py::class_<LeakyReLULayer, Layer, std::shared_ptr<LeakyReLULayer>>(
      m, "LeakyReLULayer")
      .def(py::init<float>(), py::arg("alpha") = 0.01f)
      .def("forward", &LeakyReLULayer::forward,
           py::return_value_policy::reference)
      .def("__call__", &LeakyReLULayer::forward,
           py::return_value_policy::reference);

  py::class_<DropoutLayer, Layer, std::shared_ptr<DropoutLayer>>(m,
                                                                 "DropoutLayer")
      .def(py::init<float>(), py::arg("p") = 0.5f)
      .def("forward", &DropoutLayer::forward,
           py::return_value_policy::reference)
      .def("__call__", &DropoutLayer::forward,
           py::return_value_policy::reference);

  py::class_<BatchNorm1dLayer, Layer, std::shared_ptr<BatchNorm1dLayer>>(
      m, "BatchNorm1dLayer")
      .def(py::init<int, float, float>(), py::arg("num_features"),
           py::arg("eps") = 1e-5f, py::arg("momentum") = 0.1f)
      .def("forward", &BatchNorm1dLayer::forward,
           py::return_value_policy::reference)
      .def("__call__", &BatchNorm1dLayer::forward,
           py::return_value_policy::reference);

  py::class_<Conv2dLayer, Layer, std::shared_ptr<Conv2dLayer>>(m, "Conv2dLayer")
      .def(py::init<int, int, int, int, int, int, int>(),
           py::arg("in_channels"), py::arg("out_channels"), py::arg("in_h"),
           py::arg("in_w"), py::arg("kernel_size"), py::arg("stride") = 1,
           py::arg("pad") = 0)
      .def("forward", &Conv2dLayer::forward, py::return_value_policy::reference)
      .def("__call__", &Conv2dLayer::forward,
           py::return_value_policy::reference);

  py::class_<JITGraph, std::shared_ptr<JITGraph>>(m, "JITGraph")
      .def(py::init<std::shared_ptr<Layer>, std::shared_ptr<Optimizer>,
                    std::shared_ptr<Loss>, std::shared_ptr<Regularizer>>(),
           py::arg("model"), py::arg("optimizer"), py::arg("loss_fn"),
           py::arg("regularizer") = nullptr)
      .def("trace_batch", &JITGraph::trace_batch, py::arg("dataloader"))
      .def("fast_loop", &JITGraph::fast_loop, py::arg("dataloader"),
           py::call_guard<py::gil_scoped_release>())
      .def("evaluate", &JITGraph::evaluate, py::arg("dataloader"),
           py::call_guard<py::gil_scoped_release>())
      .def("fast_fit", &JITGraph::fast_fit, py::arg("dataloader"),
           py::arg("val_dataloader") = nullptr, py::arg("epochs"),
           py::arg("tol") = 1e-4f, py::arg("n_iter_no_change") = 10,
           py::arg("verbose") = true, py::call_guard<py::gil_scoped_release>())
      .def("save_checkpoint", &JITGraph::save_checkpoint,
           py::arg("base_filepath"))
      .def("load_checkpoint", &JITGraph::load_checkpoint,
           py::arg("base_filepath"))
      .def("set_scheduler", &JITGraph::set_scheduler, py::arg("scheduler"));
}