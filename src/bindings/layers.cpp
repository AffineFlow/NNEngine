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
#include "layers/FlattenLayer.hpp"
#include "layers/LeakyReLULayer.hpp"
#include "layers/Pool2dLayer.hpp"
#include "layers/ReLULayer.hpp"

namespace py = pybind11;
using namespace affineflow::nn::core;
using namespace affineflow::nn::layers;

class PyModule : public Module {
 public:
  using Module::Module;
  ~PyModule() override = default;
  affineflow::nn::autograd::Tensor* forward(
      affineflow::nn::autograd::Tensor* input) override {
    PYBIND11_OVERRIDE_PURE(affineflow::nn::autograd::Tensor*, Module, forward,
                           input);
  }
};

void bind_layers(py::module_& m) {
  py::class_<Layer, std::shared_ptr<Layer>>(
      m, "Layer",
      "Abstract building block for differentiable model components.")
      .def("parameters", &Layer::parameters,
           "Return mutable pointers to trainable parameters.",
           py::return_value_policy::reference)
      .def("forward", &Layer::forward, py::arg("input"),
           "Run the layer on tape-owned input storage.",
           py::return_value_policy::reference)
      .def("__call__", &Layer::forward, py::arg("input"), "Alias for forward.",
           py::return_value_policy::reference)
      .def("train", &Layer::train, py::arg("mode") = true,
           "Set the layer to training mode (enables dropout, batchnorm "
           "tracking, etc).")
      .def("eval", &Layer::eval, "Set the layer to evaluation mode.");

  py::class_<Module, Layer, PyModule, std::shared_ptr<Module>>(
      m, "Module", "Base class for composing layers into a network.")
      .def(py::init<>())
      .def("forward", &Module::forward, py::arg("input"),
           "Execute the forward pass of the module.",
           py::return_value_policy::reference)
      .def("__call__", &Module::forward, py::arg("input"), "Alias for forward.",
           py::return_value_policy::reference)
      .def("predict", &Module::predict, py::arg("X"),
           "Run a detached, GIL-released inference pass.",
           py::call_guard<py::gil_scoped_release>())
      .def("parameters", &Module::parameters,
           "Return all registered parameters recursively.",
           py::return_value_policy::reference)
      .def("named_parameters", &Module::named_parameters,
           "Return a dictionary mapping parameter names to Tensors.",
           py::return_value_policy::reference_internal)
      .def("save_weights", &Module::save_weights, py::arg("filepath"),
           "Serialize module weights to disk.",
           py::call_guard<py::gil_scoped_release>())
      .def("load_weights", &Module::load_weights, py::arg("filepath"),
           "Load serialized weights from disk.",
           py::call_guard<py::gil_scoped_release>())
      .def("register_module", &Module::register_module, py::arg("name"),
           py::arg("layer"), "Register a child layer or module natively.",
           py::return_value_policy::reference);

  py::class_<DenseLayer, Layer, std::shared_ptr<DenseLayer>>(
      m, "DenseLayer",
      "Fully connected affine layer with learned weights and bias.")
      .def(py::init<int, int>(), py::arg("input_dim"), py::arg("output_dim"),
           "Initialize with Glorot uniform weights.")
      .def("forward", &DenseLayer::forward, py::arg("input"),
           "Apply the affine transform.", py::return_value_policy::reference)
      .def("__call__", &DenseLayer::forward, py::arg("input"),
           "Alias for forward.", py::return_value_policy::reference);

  py::class_<ReLULayer, Layer, std::shared_ptr<ReLULayer>>(
      m, "ReLULayer", "Elementwise rectified linear activation.")
      .def(py::init<>())
      .def("forward", &ReLULayer::forward, py::arg("input"), "Apply ReLU.",
           py::return_value_policy::reference)
      .def("__call__", &ReLULayer::forward, py::arg("input"),
           "Alias for forward.", py::return_value_policy::reference);

  py::class_<LeakyReLULayer, Layer, std::shared_ptr<LeakyReLULayer>>(
      m, "LeakyReLULayer", "Elementwise leaky rectified linear activation.")
      .def(py::init<float>(), py::arg("alpha") = 0.01f,
           "Construct with a given negative slope.")
      .def("forward", &LeakyReLULayer::forward, py::arg("input"),
           "Apply Leaky ReLU.", py::return_value_policy::reference)
      .def("__call__", &LeakyReLULayer::forward, py::arg("input"),
           "Alias for forward.", py::return_value_policy::reference);

  py::class_<DropoutLayer, Layer, std::shared_ptr<DropoutLayer>>(
      m, "DropoutLayer",
      "Randomly zeroes elements of the input tensor during training.")
      .def(py::init<float>(), py::arg("p") = 0.5f,
           "Probability of an element to be zeroed.")
      .def("forward", &DropoutLayer::forward, py::arg("input"),
           "Apply dropout mask.", py::return_value_policy::reference)
      .def("__call__", &DropoutLayer::forward, py::arg("input"),
           "Alias for forward.", py::return_value_policy::reference);

  py::class_<BatchNorm1dLayer, Layer, std::shared_ptr<BatchNorm1dLayer>>(
      m, "BatchNorm1dLayer",
      "Applies Batch Normalization over a 1D input tensor.")
      .def(py::init<int, float, float>(), py::arg("num_features"),
           py::arg("eps") = 1e-5f, py::arg("momentum") = 0.1f,
           "Initialize batch normalization.")
      .def("forward", &BatchNorm1dLayer::forward, py::arg("input"),
           "Apply batch normalization.", py::return_value_policy::reference)
      .def("__call__", &BatchNorm1dLayer::forward, py::arg("input"),
           "Alias for forward.", py::return_value_policy::reference);

  py::class_<Conv2dLayer, Layer, std::shared_ptr<Conv2dLayer>>(
      m, "Conv2dLayer", "Applies a 2D convolution over an input signal.")
      .def(py::init<int, int, int, int, int, int, int>(),
           py::arg("in_channels"), py::arg("out_channels"), py::arg("in_h"),
           py::arg("in_w"), py::arg("kernel_size"), py::arg("stride") = 1,
           py::arg("pad") = 0, "Initialize spatial convolution.")
      .def("forward", &Conv2dLayer::forward, py::arg("input"),
           "Apply 2D convolution.", py::return_value_policy::reference)
      .def("__call__", &Conv2dLayer::forward, py::arg("input"),
           "Alias for forward.", py::return_value_policy::reference);

  py::class_<JITGraph, std::shared_ptr<JITGraph>>(
      m, "JITGraph", "High-performance compiled training loop executor.")
      .def(py::init<std::shared_ptr<Layer>, std::shared_ptr<Optimizer>,
                    std::shared_ptr<Loss>, std::shared_ptr<Regularizer>>(),
           py::arg("model"), py::arg("optimizer"), py::arg("loss_fn"),
           py::arg("regularizer") = nullptr,
           "Bind the model, optimizer, and loss function to the JIT compiler.")
      .def("trace_batch", &JITGraph::trace_batch, py::arg("dataloader"),
           "Trace the autograd graph for a single batch.")
      .def("fast_loop", &JITGraph::fast_loop, py::arg("dataloader"),
           "Run a full epoch natively.",
           py::call_guard<py::gil_scoped_release>())
      .def("evaluate", &JITGraph::evaluate, py::arg("dataloader"),
           "Evaluate the model without accumulating gradients.",
           py::call_guard<py::gil_scoped_release>())
      .def("fast_fit", &JITGraph::fast_fit, py::arg("dataloader"),
           py::arg("val_dataloader"), py::arg("epochs"), py::arg("tol") = 1e-4f,
           py::arg("n_iter_no_change") = 10, py::arg("verbose") = true,
           "Train the JIT graph natively with early stopping.",
           py::call_guard<py::gil_scoped_release>())
      .def("save_checkpoint", &JITGraph::save_checkpoint,
           py::arg("base_filepath"), "Save engine state to disk.")
      .def("load_checkpoint", &JITGraph::load_checkpoint,
           py::arg("base_filepath"), "Load engine state from disk.")
      .def("set_scheduler", &JITGraph::set_scheduler, py::arg("scheduler"),
           "Attach a learning rate scheduler to the compiled graph.");

  py::class_<MaxPool2dLayer, Layer, std::shared_ptr<MaxPool2dLayer>>(
      m, "MaxPool2dLayer", "Applies a 2D max pooling over an input signal.")
      .def(py::init<int, int, int, int, int, int>(), py::arg("channels"),
           py::arg("in_h"), py::arg("in_w"), py::arg("kernel_size"),
           py::arg("stride") = 2, py::arg("pad") = 0,
           "Initialize spatial max pooling.")
      .def("forward", &MaxPool2dLayer::forward, py::arg("input"),
           "Apply 2D max pooling.", py::return_value_policy::reference)
      .def("__call__", &MaxPool2dLayer::forward, py::arg("input"),
           "Alias for forward.", py::return_value_policy::reference);

  py::class_<AvgPool2dLayer, Layer, std::shared_ptr<AvgPool2dLayer>>(
      m, "AvgPool2dLayer", "Applies a 2D average pooling over an input signal.")
      .def(py::init<int, int, int, int, int, int>(), py::arg("channels"),
           py::arg("in_h"), py::arg("in_w"), py::arg("kernel_size"),
           py::arg("stride") = 2, py::arg("pad") = 0,
           "Initialize spatial average pooling.")
      .def("forward", &AvgPool2dLayer::forward, py::arg("input"),
           "Apply 2D average pooling.", py::return_value_policy::reference)
      .def("__call__", &AvgPool2dLayer::forward, py::arg("input"),
           "Alias for forward.", py::return_value_policy::reference);

  py::class_<FlattenLayer, Layer, std::shared_ptr<FlattenLayer>>(
      m, "FlattenLayer", "Flattens a contiguous range of dims into a tensor.")
      .def(py::init<>())
      .def("forward", &FlattenLayer::forward, py::arg("input"),
           "Apply flatten.", py::return_value_policy::reference)
      .def("__call__", &FlattenLayer::forward, py::arg("input"),
           "Alias for forward.", py::return_value_policy::reference);
}