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
  m.def("set_seed", &set_seed, py::arg("seed"), R"pbdoc(
Seed the shared RNGs to make initialization, shuffling, and dropout repeatable.

Args:
  seed: Seed value to apply to the global generators.
)pbdoc");

  py::class_<mlengine::autograd::Tensor>(m, "Tensor", R"pbdoc(
Dense value-and-gradient container used by the autograd engine.

Tensors store the forward value in row-major Eigen storage and, when
gradients are enabled, a matching gradient buffer that is accumulated during
backpropagation.
)pbdoc")
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
             std::shared_ptr<mlengine::autograd::Op>>(m, "Op", R"pbdoc(
Base interface for a differentiable primitive recorded on the tape.

Ops encapsulate a forward computation together with its adjoint so the tape
can replay the graph without heap allocations in the training loop.
)pbdoc")
      .def(py::init<>())
      .def("forward", &mlengine::autograd::Op::forward, R"pbdoc(
Execute the forward pass for the primitive.
)pbdoc")
      .def("backward", &mlengine::autograd::Op::backward, R"pbdoc(
Accumulate gradients for the primitive inputs.
)pbdoc");

  py::class_<mlengine::autograd::Tape,
             std::shared_ptr<mlengine::autograd::Tape>>(m, "Tape", R"pbdoc(
Arena-style memory allocator and operation replay log.

The tape owns all transient tensors it allocates. Callers receive raw
pointers for convenience, but the lifetime is tied to the tape and must not
be managed manually.
)pbdoc")
      .def(py::init<bool>(), py::arg("record_ops") = true, R"pbdoc(
Construct a tape that optionally records ops for replay.

Args:
  record_ops: Whether ops should be stored for backward replay.
)pbdoc")
      .def_property(
          "record_ops",
          [](const mlengine::autograd::Tape& t) { return t.record_ops_; },
          [](mlengine::autograd::Tape& t, bool value) {
            t.record_ops_ = value;
          })
      .def("alloc_tensor", &mlengine::autograd::Tape::alloc_tensor,
           py::arg("rows"), py::arg("cols"), py::arg("requires_grad") = true,
           py::return_value_policy::reference, R"pbdoc(
Allocate a tensor from the tape's arena.

Args:
  rows: Number of rows.
  cols: Number of columns.
  requires_grad: Whether the tensor participates in autograd.

Returns:
  Pointer to the tape-owned tensor.
)pbdoc")
      .def("push_tensor", &mlengine::autograd::Tape::push_tensor,
           py::arg("data"), py::arg("requires_grad") = true,
           py::return_value_policy::reference, R"pbdoc(
Copy a dense matrix into tape-owned storage.

Args:
  data: Matrix value to push.
  requires_grad: Whether the resulting tensor requires gradients.

Returns:
  Pointer to the tape-owned tensor.
)pbdoc")
      .def("record_op", &mlengine::autograd::Tape::record_op, py::arg("op"),
           R"pbdoc(
Record an op for future replay if recording is enabled.
)pbdoc")
      .def("backward", &mlengine::autograd::Tape::backward, R"pbdoc(
Alias for replaying the backward pass.
)pbdoc")
      .def("reset", &mlengine::autograd::Tape::reset, R"pbdoc(
Clear the recorded graph and rewind arena allocation.
)pbdoc");

  py::class_<Layer, std::shared_ptr<Layer>>(m, "Layer", R"pbdoc(
Abstract building block for differentiable model components.
)pbdoc")
      .def("parameters", &Layer::parameters, py::return_value_policy::reference,
           R"pbdoc(
Return mutable pointers to trainable parameters.
)pbdoc")
      .def("forward", &Layer::forward, py::return_value_policy::reference,
           R"pbdoc(
Run the layer on tape-owned input storage.
)pbdoc")
      .def("__call__", &Layer::forward, py::return_value_policy::reference)
      .def("train", &Layer::train, py::arg("mode") = true, R"pbdoc(
Set the training mode for this layer.
)pbdoc")
      .def("eval", &Layer::eval, R"pbdoc(
Set the evaluation mode for this layer.
)pbdoc");

  py::class_<Module, Layer, PyModule, std::shared_ptr<Module>>(m, "Module",
                                                               R"pbdoc(
Composite layer that manages submodules and persistence utilities.
)pbdoc")
      .def(py::init<>())
      .def("forward", &Module::forward, py::return_value_policy::reference,
           R"pbdoc(
Run the layer on tape-owned input storage.
)pbdoc")
      .def("__call__", &Module::forward, py::return_value_policy::reference)
      .def("predict", &Module::predict, py::arg("X"),
           py::call_guard<py::gil_scoped_release>(), R"pbdoc(
Execute a forward pass without tracking gradients.

Args:
  X: Input matrix.

Returns:
  The predicted output matrix.
)pbdoc")
      .def("parameters", &Module::parameters,
           py::return_value_policy::reference, R"pbdoc(
Collect all trainable tensors from child modules.

Returns:
  Flat list of parameter tensors.
)pbdoc")
      .def("named_parameters", &Module::named_parameters,
           py::return_value_policy::reference_internal, R"pbdoc(
Collect recursively prefixed named parameters.

Returns:
  Key-value map of parameter strings to tensors.
)pbdoc")
      .def("save_weights", &Module::save_weights, py::arg("filepath"),
           py::call_guard<py::gil_scoped_release>(), R"pbdoc(
Serialize model parameters to a binary .nne file.

Args:
  filepath: Target file destination.
)pbdoc")
      .def("load_weights", &Module::load_weights, py::arg("filepath"),
           py::call_guard<py::gil_scoped_release>(), R"pbdoc(
Load and map parameters from a binary .nne file robustly.

Args:
  filepath: Source file.
)pbdoc")
      .def("register_module", &Module::register_module, py::arg("name"),
           py::arg("layer"), py::return_value_policy::reference, R"pbdoc(
Register a child layer with a specific string identifier.

Args:
  name: The identifier for the layer (e.g., "fc1").
  layer: Layer to append to the module hierarchy.

Returns:
  The same layer pointer for fluent composition.
)pbdoc");

  py::class_<DataLoader>(m, "DataLoader", R"pbdoc(
Mini-batch iterator over dense feature and target matrices.

Handles zero-copy slicing of the dataset and optional epoch reshuffling.
)pbdoc")
      .def(py::init<const mlengine::MatrixRM&, const mlengine::MatrixRM&,
                    size_t, bool, bool>(),
           py::arg("X"), py::arg("y"), py::arg("batch_size"),
           py::arg("shuffle") = true, py::arg("drop_last") = false, R"pbdoc(
Build a loader from in-memory feature and target matrices.

Args:
  X: Feature matrix with samples in rows.
  y: Target matrix with matching row count.
  batch_size: Number of samples per batch.
  shuffle: Whether to shuffle rows at the start of each epoch.
  drop_last: Whether to drop the final partial batch.
)pbdoc")
      .def("reset", &DataLoader::reset, R"pbdoc(
Reset iteration state and reshuffle if enabled.
)pbdoc");
}

void bind_losses_and_regs(py::module_& m) {
  py::class_<Loss, std::shared_ptr<Loss>>(m, "Loss", R"pbdoc(
Objective function that produces a scalar training signal.
)pbdoc")
      .def("forward", &Loss::forward, py::arg("predictions"),
           py::arg("targets"), R"pbdoc(
Bind tensors and compute the forward loss.
)pbdoc");

  py::class_<MSELoss, Loss, std::shared_ptr<MSELoss>>(m, "MSELoss", R"pbdoc(
Mean-squared-error objective for regression.
)pbdoc")
      .def(py::init<>());

  py::class_<SoftmaxCrossEntropyLoss, Loss,
             std::shared_ptr<SoftmaxCrossEntropyLoss>>(
      m, "SoftmaxCrossEntropyLoss", R"pbdoc(
Numerically stable softmax cross-entropy for classification.
)pbdoc")
      .def(py::init<>());

  py::class_<Regularizer, std::shared_ptr<Regularizer>>(m, "Regularizer",
                                                        R"pbdoc(
Penalty term applied to trainable parameters during optimization.
)pbdoc");

  py::class_<L2Regularizer, Regularizer, std::shared_ptr<L2Regularizer>>(
      m, "L2Regularizer", R"pbdoc(
L2 weight decay regularizer.
)pbdoc")
      .def(py::init<float>(), py::arg("l2") = 0.0001f, R"pbdoc(
Construct the regularizer with the desired penalty strength.

Args:
  l2: Coefficient applied to the squared-norm penalty.
)pbdoc");
}

void bind_optimizers(py::module_& m) {
  py::class_<Optimizer, std::shared_ptr<Optimizer>>(m, "Optimizer", R"pbdoc(
Base class for parameter update rules.
)pbdoc");

  py::class_<SGD, Optimizer, std::shared_ptr<SGD>>(m, "SGD", R"pbdoc(
Plain stochastic gradient descent optimizer.
)pbdoc")
      .def(py::init<float>(), py::arg("learning_rate") = 0.01f, R"pbdoc(
Construct SGD with the supplied learning rate.

Args:
  learning_rate: Step size used for parameter updates.
)pbdoc");

  py::class_<Adam, Optimizer, std::shared_ptr<Adam>>(m, "Adam", R"pbdoc(
Adam optimizer with first/second moment estimates.
)pbdoc")
      .def(py::init<float>(), py::arg("learning_rate") = 0.001f, R"pbdoc(
Construct Adam with the supplied learning rate.

Args:
  learning_rate: Base step size used for updates.
)pbdoc");

  py::class_<Scheduler, std::shared_ptr<Scheduler>>(m, "Scheduler", R"pbdoc(
Base class for learning rate scheduling.
)pbdoc");

  py::class_<StepLR, Scheduler, std::shared_ptr<StepLR>>(m, "StepLR", R"pbdoc(
Decays the learning rate of each parameter group by gamma every step_size epochs.
)pbdoc")
      .def(py::init<std::shared_ptr<Optimizer>, int, float>(),
           py::arg("optimizer"), py::arg("step_size"), py::arg("gamma") = 0.1f)
      .def("step", &StepLR::step);
}

void bind_layers(py::module_& m) {
  py::class_<DenseLayer, Layer, std::shared_ptr<DenseLayer>>(m, "DenseLayer",
                                                             R"pbdoc(
Fully connected affine layer with learned weights and bias.
)pbdoc")
      .def(py::init<int, int>(), R"pbdoc(
Create a dense layer with Glorot-style uniform initialization.

Args:
  input_dim: Number of input features.
  output_dim: Number of output features.
)pbdoc")
      .def("forward", &DenseLayer::forward, py::return_value_policy::reference,
           R"pbdoc(
Apply the affine transform and record the corresponding tape ops.
)pbdoc")
      .def("__call__", &DenseLayer::forward, py::return_value_policy::reference)
      .def("get_weights", &DenseLayer::get_weights,
           R"pbdoc(Access the current weight matrix.)pbdoc")
      .def("get_bias", &DenseLayer::get_bias,
           R"pbdoc(Access the current bias row vector.)pbdoc");

  py::class_<ReLULayer, Layer, std::shared_ptr<ReLULayer>>(m, "ReLULayer",
                                                           R"pbdoc(
Elementwise rectified linear activation.
)pbdoc")
      .def(py::init<>())
      .def("forward", &ReLULayer::forward, py::return_value_policy::reference,
           R"pbdoc(
Apply ReLU to the incoming activation tensor.
)pbdoc")
      .def("__call__", &ReLULayer::forward, py::return_value_policy::reference);

  py::class_<LeakyReLULayer, Layer, std::shared_ptr<LeakyReLULayer>>(
      m, "LeakyReLULayer", R"pbdoc(
Elementwise leaky rectified linear activation.
)pbdoc")
      .def(py::init<float>(), py::arg("alpha") = 0.01f, R"pbdoc(
Construct a leaky ReLU layer with the given negative slope.

Args:
  alpha: Slope applied to negative activations.
)pbdoc")
      .def("forward", &LeakyReLULayer::forward,
           py::return_value_policy::reference, R"pbdoc(
Apply leaky ReLU to the incoming activation tensor.
)pbdoc")
      .def("__call__", &LeakyReLULayer::forward,
           py::return_value_policy::reference);

  py::class_<mlengine::layers::DropoutLayer, Layer,
             std::shared_ptr<mlengine::layers::DropoutLayer>>(m, "DropoutLayer",
                                                              R"pbdoc(
Randomly zeroes some of the elements of the input tensor.
)pbdoc")
      .def(py::init<float>(), py::arg("p") = 0.5f, R"pbdoc(
Create a Dropout layer.

Args:
  p: Probability of an element to be zeroed. Default: 0.5
)pbdoc")
      .def("forward", &mlengine::layers::DropoutLayer::forward,
           py::return_value_policy::reference)
      .def("__call__", &mlengine::layers::DropoutLayer::forward,
           py::return_value_policy::reference);

  py::class_<mlengine::layers::BatchNorm1dLayer, Layer,
             std::shared_ptr<mlengine::layers::BatchNorm1dLayer>>(
      m, "BatchNorm1dLayer", R"pbdoc(
Applies Batch Normalization over a 1D input tensor.
)pbdoc")
      .def(py::init<int, float, float>(), py::arg("num_features"),
           py::arg("eps") = 1e-5f, py::arg("momentum") = 0.1f, R"pbdoc(
Construct a Batch Normalization layer.

Args:
  num_features: Number of features in the input.
  eps: Value added to denominator for numerical stability.
  momentum: Value used for running mean and variance computation.
)pbdoc")
      .def("forward", &mlengine::layers::BatchNorm1dLayer::forward,
           py::return_value_policy::reference)
      .def("__call__", &mlengine::layers::BatchNorm1dLayer::forward,
           py::return_value_policy::reference);

  py::class_<mlengine::layers::Conv2dLayer, Layer,
             std::shared_ptr<mlengine::layers::Conv2dLayer>>(m, "Conv2dLayer",
                                                             R"pbdoc(
Applies a 2D convolution over an input signal.
)pbdoc")
      .def(py::init<int, int, int, int, int, int, int>(),
           py::arg("in_channels"), py::arg("out_channels"), py::arg("in_h"),
           py::arg("in_w"), py::arg("kernel_size"), py::arg("stride") = 1,
           py::arg("pad") = 0, R"pbdoc(
Construct a 2D Convolutional Layer.
)pbdoc")
      .def("forward", &mlengine::layers::Conv2dLayer::forward,
           py::return_value_policy::reference)
      .def("__call__", &mlengine::layers::Conv2dLayer::forward,
           py::return_value_policy::reference);
}

void bind_model(py::module_& m) {
  py::class_<JITGraph, std::shared_ptr<JITGraph>>(m, "JITGraph", R"pbdoc(
Just-In-Time compiled training loop executor.

Traces a computational graph during the first batch and executes
a highly optimized, zero-allocation native C++ replay loop for all subsequent batches.
)pbdoc")
      .def(py::init<std::shared_ptr<Layer>, std::shared_ptr<Optimizer>,
                    std::shared_ptr<Loss>, std::shared_ptr<Regularizer>>(),
           py::arg("model"), py::arg("optimizer"), py::arg("loss_fn"),
           py::arg("regularizer") = nullptr)
      .def("trace_batch", &JITGraph::trace_batch, py::arg("dataloader"),
           R"pbdoc(
Trace the computational graph for the first batch.
)pbdoc")
      .def("fast_loop", &JITGraph::fast_loop, py::arg("dataloader"),
           py::call_guard<py::gil_scoped_release>(), R"pbdoc(
Replay the traced graph natively across remaining batches.
)pbdoc")
      .def("evaluate", &JITGraph::evaluate, py::arg("dataloader"),
           py::call_guard<py::gil_scoped_release>(), R"pbdoc(
Run an evaluation pass without updating parameters.
)pbdoc")
      .def("fast_fit", &JITGraph::fast_fit, py::arg("dataloader"),
           py::arg("val_dataloader") = nullptr, py::arg("epochs"),
           py::arg("tol") = 1e-4f, py::arg("n_iter_no_change") = 10,
           py::arg("verbose") = true, py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
Execute the full multi-epoch JIT training loop.
)pbdoc")
      .def("save_checkpoint", &JITGraph::save_checkpoint,
           py::arg("base_filepath"), R"pbdoc(
Native serialization for weights and optimizer moments.
)pbdoc")
      .def("load_checkpoint", &JITGraph::load_checkpoint,
           py::arg("base_filepath"), R"pbdoc(
Native deserialization for weights and optimizer moments.
)pbdoc")
      .def("set_scheduler", &JITGraph::set_scheduler, py::arg("scheduler"),
           R"pbdoc(
Attach a learning rate scheduler to the training loop.
)pbdoc");
}

PYBIND11_MODULE(nn_core, m) {
  m.doc() = "C++ Core Engine for NNEngine";
  bind_core_utils(m);
  bind_losses_and_regs(m);
  bind_optimizers(m);
  bind_layers(m);
  bind_model(m);
}