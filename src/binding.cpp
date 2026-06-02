#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <sstream>

#include "autograd/Op.hpp"
#include "autograd/Tape.hpp"
#include "autograd/Tensor.hpp"
#include "autograd/ops/PyOp.hpp"
#include "core/DataLoader.hpp"
#include "core/JITGraph.hpp"
#include "core/Layer.hpp"
#include "core/Loss.hpp"
#include "core/Module.hpp"
#include "core/Optimizer.hpp"
#include "core/Random.hpp"
#include "core/Regularizer.hpp"
#include "core/Scheduler.hpp"
#include "layers/BatchNorm1dLayer.hpp"
#include "layers/Conv2dLayer.hpp"
#include "layers/DenseLayer.hpp"
#include "layers/DropoutLayer.hpp"
#include "layers/LeakyReLULayer.hpp"
#include "layers/ReLULayer.hpp"
#include "losses/MSELoss.hpp"
#include "losses/SoftmaxCrossEntropyLoss.hpp"
#include "optimizers/Adam.hpp"
#include "optimizers/SGD.hpp"
#include "regularizers/L2Regularizer.hpp"

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

void bind_core_utils(py::module_& m) {
  m.def("set_seed", &set_seed, py::arg("seed"));

  py::class_<mlengine::autograd::Tensor>(m, "Tensor")
      .def_property(
          "data",
          [](mlengine::autograd::Tensor& t) -> Eigen::Ref<mlengine::MatrixRM> {
            return t.data;
          },
          [](mlengine::autograd::Tensor& t, const mlengine::MatrixRM& v) {
            t.data = v;
          })
      .def_property(
          "grad",
          [](mlengine::autograd::Tensor& t) -> Eigen::Ref<mlengine::MatrixRM> {
            return t.grad;
          },
          [](mlengine::autograd::Tensor& t, const mlengine::MatrixRM& v) {
            t.grad = v;
          })
      .def_property(
          "requires_grad",
          [](const mlengine::autograd::Tensor& t) { return t.requires_grad; },
          [](mlengine::autograd::Tensor& t, bool value) {
            t.requires_grad = value;
          })
      .def("__repr__", [](const mlengine::autograd::Tensor& t) {
        std::ostringstream oss;
        oss << "Tensor(shape=(" << t.data.rows() << ", " << t.data.cols()
            << "), requires_grad=" << (t.requires_grad ? "True" : "False")
            << ")\n"
            << t.data;
        return oss.str();
      });

  py::class_<mlengine::autograd::Op, mlengine::autograd::ops::PyOp,
             std::shared_ptr<mlengine::autograd::Op>>(m, "Op")
      .def(py::init<>())
      .def("forward", &mlengine::autograd::Op::forward)
      .def("backward", &mlengine::autograd::Op::backward);

  py::class_<mlengine::autograd::Tape,
             std::shared_ptr<mlengine::autograd::Tape>>(m, "Tape")
      .def(py::init<bool>(), py::arg("record_ops") = true)
      .def_property(
          "record_ops",
          [](const mlengine::autograd::Tape& t) { return t.record_ops_; },
          [](mlengine::autograd::Tape& t, bool value) {
            t.record_ops_ = value;
          })
      .def("alloc_tensor", &mlengine::autograd::Tape::alloc_tensor,
           py::arg("rows"), py::arg("cols"), py::arg("requires_grad") = true,
           py::return_value_policy::reference)
      .def("push_tensor", &mlengine::autograd::Tape::push_tensor,
           py::arg("data"), py::arg("requires_grad") = true,
           py::return_value_policy::reference)
      .def("record_op", &mlengine::autograd::Tape::record_op, py::arg("op"))
      .def("backward", &mlengine::autograd::Tape::backward)
      .def("reset", &mlengine::autograd::Tape::reset);

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

  py::class_<DataLoader>(m, "DataLoader")
      .def(py::init<const mlengine::MatrixRM&, const mlengine::MatrixRM&,
                    size_t, bool, bool>(),
           py::arg("X"), py::arg("y"), py::arg("batch_size"),
           py::arg("shuffle") = true, py::arg("drop_last") = false)
      .def("reset", &DataLoader::reset);
}

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

void bind_optimizers(py::module_& m) {
  py::class_<Optimizer, std::shared_ptr<Optimizer>>(m, "Optimizer");

  py::class_<SGD, Optimizer, std::shared_ptr<SGD>>(m, "SGD").def(
      py::init<float>(), py::arg("learning_rate") = 0.01f);

  py::class_<Adam, Optimizer, std::shared_ptr<Adam>>(m, "Adam").def(
      py::init<float>(), py::arg("learning_rate") = 0.001f);

  py::class_<Scheduler, std::shared_ptr<Scheduler>>(m, "Scheduler");

  py::class_<StepLR, Scheduler, std::shared_ptr<StepLR>>(m, "StepLR")
      .def(py::init<std::shared_ptr<Optimizer>, int, float>(),
           py::arg("optimizer"), py::arg("step_size"), py::arg("gamma") = 0.1f)
      .def("step", &StepLR::step);
}

void bind_layers(py::module_& m) {
  py::class_<DenseLayer, Layer, std::shared_ptr<DenseLayer>>(m, "DenseLayer")
      .def(py::init<int, int>())
      .def("forward", &DenseLayer::forward, py::return_value_policy::reference)
      .def("__call__", &DenseLayer::forward, py::return_value_policy::reference)
      .def("get_weights", &DenseLayer::get_weights)
      .def("get_bias", &DenseLayer::get_bias);

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

  py::class_<mlengine::layers::DropoutLayer, Layer,
             std::shared_ptr<mlengine::layers::DropoutLayer>>(m, "DropoutLayer")
      .def(py::init<float>(), py::arg("p") = 0.5f)
      .def("forward", &mlengine::layers::DropoutLayer::forward,
           py::return_value_policy::reference)
      .def("__call__", &mlengine::layers::DropoutLayer::forward,
           py::return_value_policy::reference);

  py::class_<mlengine::layers::BatchNorm1dLayer, Layer,
             std::shared_ptr<mlengine::layers::BatchNorm1dLayer>>(
      m, "BatchNorm1dLayer")
      .def(py::init<int, float, float>(), py::arg("num_features"),
           py::arg("eps") = 1e-5f, py::arg("momentum") = 0.1f)
      .def("forward", &mlengine::layers::BatchNorm1dLayer::forward,
           py::return_value_policy::reference)
      .def("__call__", &mlengine::layers::BatchNorm1dLayer::forward,
           py::return_value_policy::reference);

  py::class_<mlengine::layers::Conv2dLayer, Layer,
             std::shared_ptr<mlengine::layers::Conv2dLayer>>(m, "Conv2dLayer")
      .def(py::init<int, int, int, int, int, int, int>(),
           py::arg("in_channels"), py::arg("out_channels"), py::arg("in_h"),
           py::arg("in_w"), py::arg("kernel_size"), py::arg("stride") = 1,
           py::arg("pad") = 0)
      .def("forward", &mlengine::layers::Conv2dLayer::forward,
           py::return_value_policy::reference)
      .def("__call__", &mlengine::layers::Conv2dLayer::forward,
           py::return_value_policy::reference);
}

void bind_model(py::module_& m) {
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

PYBIND11_MODULE(nn_core, m) {
  m.doc() = "C++ Core Engine for NNEngine";
  bind_core_utils(m);
  bind_losses_and_regs(m);
  bind_optimizers(m);
  bind_layers(m);
  bind_model(m);
}