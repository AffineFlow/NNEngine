#include <pybind11/eigen.h>
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
#include "layers/DenseLayer.hpp"
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
Seed the shared engine RNG.

Args:
  seed: Seed value used for layer initialization and dataloader shuffling.
)pbdoc");

  py::class_<mlengine::autograd::Tensor>(m, "Tensor", R"pbdoc(
Dense value-and-gradient container exposed to Python as float32 NumPy arrays.

The data and grad properties are backed by Eigen row-major storage and map to
NumPy ndarray objects with dtype float32.
)pbdoc")
      .def_property(
          "data",
          [](mlengine::autograd::Tensor& t) -> Eigen::Ref<mlengine::MatrixRM> {
            return t.data;
          },
          [](mlengine::autograd::Tensor& t, const mlengine::MatrixRM& v) {
            t.data = v;
          },
          R"pbdoc(
Forward value as a float32 NumPy ndarray with row-major layout.

Returns:
  A view-compatible ndarray backed by Eigen storage.
)pbdoc")
      .def_property(
          "grad",
          [](mlengine::autograd::Tensor& t) -> Eigen::Ref<mlengine::MatrixRM> {
            return t.grad;
          },
          [](mlengine::autograd::Tensor& t, const mlengine::MatrixRM& v) {
            t.grad = v;
          },
          R"pbdoc(
Gradient buffer as a float32 NumPy ndarray with row-major layout.

Returns:
  A view-compatible ndarray backed by Eigen storage.
)pbdoc")
      .def_property(
          "requires_grad",
          [](const mlengine::autograd::Tensor& t) { return t.requires_grad; },
          [](mlengine::autograd::Tensor& t, bool value) {
            t.requires_grad = value;
          },
          R"pbdoc(
Whether the tensor participates in gradient recording.
)pbdoc")
      .def(
          "__repr__",
          [](const mlengine::autograd::Tensor& t) {
            std::ostringstream oss;
            oss << "Tensor(shape=(" << t.data.rows() << ", " << t.data.cols()
                << "), requires_grad=" << (t.requires_grad ? "True" : "False")
                << ")\n"
                << t.data;
            return oss.str();
          },
          R"pbdoc(Return a compact debugging representation.)pbdoc");

  py::class_<mlengine::autograd::Op, mlengine::autograd::ops::PyOp,
             std::shared_ptr<mlengine::autograd::Op>>(m, "Op", R"pbdoc(
Base class for custom differentiable primitives.

Python users can subclass Op to define custom forward and backward passes.
)pbdoc")
      .def(py::init<>())
      .def("forward", &mlengine::autograd::Op::forward, R"pbdoc(
Execute the forward pass for the primitive.
)pbdoc")
      .def("backward", &mlengine::autograd::Op::backward, R"pbdoc(
Execute the backward pass for the primitive.
)pbdoc");

  py::class_<mlengine::autograd::Tape,
             std::shared_ptr<mlengine::autograd::Tape>>(m, "Tape", R"pbdoc(
Arena allocator and replay log for autograd tensors and ops.

Allocated tensors are owned by the tape and reused across replayed batches.
)pbdoc")
      .def(py::init<bool>(), py::arg("record_ops") = true, R"pbdoc(
Create a tape that optionally records ops for replay.

Args:
  record_ops: Whether to store ops for backward replay.
)pbdoc")
      .def_property(
          "record_ops",
          [](const mlengine::autograd::Tape& t) { return t.record_ops_; },
          [](mlengine::autograd::Tape& t, bool value) {
            t.record_ops_ = value;
          },
          R"pbdoc(Whether ops are recorded for replay.)pbdoc")
      .def("alloc_tensor", &mlengine::autograd::Tape::alloc_tensor,
           py::arg("rows"), py::arg("cols"), py::arg("requires_grad") = true,
           py::return_value_policy::reference, R"pbdoc(
Allocate a tensor from tape-owned storage.

Args:
  rows: Number of rows.
  cols: Number of columns.
  requires_grad: Whether the tensor participates in autograd.

Returns:
  A Tensor owned by the tape, not the caller.
)pbdoc")
      .def("push_tensor", &mlengine::autograd::Tape::push_tensor,
           py::arg("data"), py::arg("requires_grad") = true,
           py::return_value_policy::reference, R"pbdoc(
Copy a dense matrix into tape-owned storage.

Args:
  data: Input matrix to materialize as a tensor.
  requires_grad: Whether the resulting tensor requires gradients.

Returns:
  A Tensor owned by the tape, not the caller.
)pbdoc")
      .def("record_op", &mlengine::autograd::Tape::record_op, py::arg("op"),
           R"pbdoc(
Append a differentiable primitive to the replay log.
)pbdoc")
      .def("backward", &mlengine::autograd::Tape::backward, R"pbdoc(
Replay the recorded backward pass.
)pbdoc")
      .def("reset", &mlengine::autograd::Tape::reset, R"pbdoc(
Clear recorded ops and rewind arena allocation.
)pbdoc");

  py::class_<Layer, std::shared_ptr<Layer>>(m, "Layer", R"pbdoc(
Abstract building block for differentiable model components.
)pbdoc")
      .def("parameters", &Layer::parameters, py::return_value_policy::reference,
           R"pbdoc(Return mutable parameter tensors owned by the layer.)pbdoc")
      .def("forward", &Layer::forward, py::return_value_policy::reference,
           R"pbdoc(
Run the layer on tape-owned input storage.

Args:
  input: Input tensor to transform.

Returns:
  A Tensor owned by the tape, not the caller.
)pbdoc")
      .def("__call__", &Layer::forward, py::return_value_policy::reference,
           R"pbdoc(Alias for forward.)pbdoc");

  py::class_<Module, Layer, PyModule, std::shared_ptr<Module>>(m, "Module",
                                                               R"pbdoc(
Composite layer with parameter collection and weight persistence helpers.
)pbdoc")
      .def(py::init<>(), R"pbdoc(Create an empty module container.)pbdoc")
      .def("forward", &Module::forward, py::return_value_policy::reference,
           R"pbdoc(Run the model's forward pass.)pbdoc")
      .def("__call__", &Module::forward, py::return_value_policy::reference,
           R"pbdoc(Alias for forward.)pbdoc")
      .def("predict", &Module::predict,
           py::call_guard<py::gil_scoped_release>(), R"pbdoc(
Run inference without recording autograd operations.

Args:
  X: Input matrix. Eigen row-major float32 storage is exposed as a NumPy ndarray.

Returns:
  A float32 NumPy ndarray containing predictions.
)pbdoc")
      .def(
          "parameters", &Module::parameters, py::return_value_policy::reference,
          R"pbdoc(Return mutable parameter tensors owned by child modules.)pbdoc")
      .def("save_weights", &Module::save_weights, py::arg("filepath"),
           py::call_guard<py::gil_scoped_release>(), R"pbdoc(
Serialize model parameters to a binary .nne file in native C++ code.

Args:
  filepath: Destination path for the serialized weights.

Notes:
  The serialization work executes in C++ with the GIL released.
)pbdoc")
      .def("load_weights", &Module::load_weights, py::arg("filepath"),
           py::call_guard<py::gil_scoped_release>(), R"pbdoc(
Load model parameters from a binary .nne file in native C++ code.

Args:
  filepath: Source path for the serialized weights.

Notes:
  The deserialization work executes in C++ with the GIL released.
)pbdoc")
      .def("add_module", &Module::add_module, py::arg("layer"),
           py::return_value_policy::reference, R"pbdoc(
Register a child layer and retain shared ownership.

Args:
  layer: Layer to append to the module hierarchy.

Returns:
  The same layer object for fluent model construction.
)pbdoc");

  py::class_<DataLoader>(m, "DataLoader", R"pbdoc(
Mini-batch iterator over dense feature and target matrices.
)pbdoc")
      .def(py::init<const mlengine::MatrixRM&, const mlengine::MatrixRM&,
                    size_t, bool, bool>(),
           py::arg("X"), py::arg("y"), py::arg("batch_size"),
           py::arg("shuffle") = true, py::arg("drop_last") = false, R"pbdoc(
Create a mini-batch loader from in-memory matrices.

Args:
  X: Feature matrix with samples in rows.
  y: Target matrix with matching row count.
  batch_size: Number of samples per batch.
  shuffle: Whether to shuffle rows at epoch boundaries.
  drop_last: Whether to drop the final partial batch.
)pbdoc")
      .def("reset", &DataLoader::reset,
           R"pbdoc(Reset iteration and reshuffle if enabled.)pbdoc");
}

void bind_losses_and_regs(py::module_& m) {
  py::class_<Loss, std::shared_ptr<Loss>>(m, "Loss", R"pbdoc(
Base class for scalar objective functions.
)pbdoc")
      .def("forward", &Loss::forward, py::arg("predictions"),
           py::arg("targets"),
           R"pbdoc(
Evaluate the loss and seed the prediction gradient.

Args:
  predictions: Model outputs.
  targets: Reference targets.

Returns:
  Scalar loss value for the current batch.
)pbdoc");
  py::class_<MSELoss, Loss, std::shared_ptr<MSELoss>>(m, "MSELoss", R"pbdoc(
Mean-squared-error loss for regression.
)pbdoc")
      .def(py::init<>(), R"pbdoc(Create an MSE loss object.)pbdoc");
  py::class_<SoftmaxCrossEntropyLoss, Loss,
             std::shared_ptr<SoftmaxCrossEntropyLoss>>(
      m, "SoftmaxCrossEntropyLoss", R"pbdoc(
Numerically stable softmax cross-entropy loss for classification.
)pbdoc")
      .def(py::init<>(),
           R"pbdoc(Create a softmax cross-entropy loss object.)pbdoc");
  py::class_<Regularizer, std::shared_ptr<Regularizer>>(m, "Regularizer",
                                                        R"pbdoc(
Base class for parameter penalties.
)pbdoc");
  py::class_<L2Regularizer, Regularizer, std::shared_ptr<L2Regularizer>>(
      m, "L2Regularizer", R"pbdoc(
L2 weight decay regularizer.
)pbdoc")
      .def(py::init<float>(), py::arg("l2") = 0.0001f, R"pbdoc(
Create an L2 regularizer.

Args:
  l2: Penalty coefficient.
)pbdoc");
}

void bind_optimizers(py::module_& m) {
  py::class_<Optimizer, std::shared_ptr<Optimizer>>(m, "Optimizer", R"pbdoc(
Base class for update rules.
)pbdoc");
  py::class_<SGD, Optimizer, std::shared_ptr<SGD>>(m, "SGD", R"pbdoc(
Plain stochastic gradient descent optimizer.
)pbdoc")
      .def(py::init<float>(), py::arg("learning_rate") = 0.01f, R"pbdoc(
Create an SGD optimizer.

Args:
  learning_rate: Step size used for updates.
)pbdoc");
  py::class_<Adam, Optimizer, std::shared_ptr<Adam>>(m, "Adam", R"pbdoc(
Adam optimizer with bias-corrected first and second moments.
)pbdoc")
      .def(py::init<float>(), py::arg("learning_rate") = 0.001f, R"pbdoc(
Create an Adam optimizer.

Args:
  learning_rate: Step size used for updates.
)pbdoc");
}

void bind_layers(py::module_& m) {
  py::class_<DenseLayer, Layer, std::shared_ptr<DenseLayer>>(m, "DenseLayer",
                                                             R"pbdoc(
Fully connected affine layer.
)pbdoc")
      .def(py::init<int, int>(), R"pbdoc(
Create a dense layer.

Args:
  input_dim: Number of input features.
  output_dim: Number of output features.
)pbdoc")
      .def("forward", &DenseLayer::forward, py::return_value_policy::reference,
           R"pbdoc(
Apply the affine transform and record the corresponding tape ops.

Args:
  input: Input activation tensor.

Returns:
  A Tensor owned by the tape, not the caller.
)pbdoc")
      .def("__call__", &DenseLayer::forward, py::return_value_policy::reference,
           R"pbdoc(Alias for forward.)pbdoc")
      .def("get_weights", &DenseLayer::get_weights, R"pbdoc(
Return the current weight matrix as a float32 NumPy ndarray.
)pbdoc")
      .def("get_bias", &DenseLayer::get_bias, R"pbdoc(
Return the current bias matrix as a float32 NumPy ndarray.
)pbdoc");

  py::class_<ReLULayer, Layer, std::shared_ptr<ReLULayer>>(m, "ReLULayer",
                                                           R"pbdoc(
Elementwise rectified linear activation layer.
)pbdoc")
      .def(py::init<>(), R"pbdoc(Create a ReLU layer.)pbdoc")
      .def("forward", &ReLULayer::forward, py::return_value_policy::reference,
           R"pbdoc(
Apply ReLU to the incoming activation tensor.

Args:
  input: Input activation tensor.

Returns:
  A Tensor owned by the tape, not the caller.
)pbdoc")
      .def("__call__", &ReLULayer::forward, py::return_value_policy::reference,
           R"pbdoc(Alias for forward.)pbdoc");

  py::class_<LeakyReLULayer, Layer, std::shared_ptr<LeakyReLULayer>>(
      m, "LeakyReLULayer", R"pbdoc(
Elementwise leaky rectified linear activation layer.
)pbdoc")
      .def(py::init<float>(), py::arg("alpha") = 0.01f, R"pbdoc(
Create a leaky ReLU layer.

Args:
  alpha: Slope applied to negative activations.
)pbdoc")
      .def("forward", &LeakyReLULayer::forward,
           py::return_value_policy::reference, R"pbdoc(
Apply leaky ReLU to the incoming activation tensor.

Args:
  input: Input activation tensor.

Returns:
  A Tensor owned by the tape, not the caller.
)pbdoc")
      .def("__call__", &LeakyReLULayer::forward,
           py::return_value_policy::reference);
}

void bind_model(py::module_& m) {
  py::class_<JITGraph, std::shared_ptr<JITGraph>>(m, "JITGraph", R"pbdoc(
JIT training loop that traces one batch and replays it efficiently.
)pbdoc")
      .def(py::init<std::shared_ptr<Layer>, std::shared_ptr<Optimizer>,
                    std::shared_ptr<Loss>, std::shared_ptr<Regularizer>>(),
           py::arg("model"), py::arg("optimizer"), py::arg("loss_fn"),
           py::arg("regularizer") = nullptr, R"pbdoc(
Create a compiled training graph.

Args:
  model: Differentiable network to optimize.
  optimizer: Update rule applied after each batch.
  loss_fn: Loss function used for supervision.
  regularizer: Optional regularization term.
)pbdoc")
      .def("trace_batch", &JITGraph::trace_batch, py::arg("dataloader"),
           R"pbdoc(
Trace one batch, record the tape, and perform the first update.

Args:
  dataloader: Batch source used for tracing.

Returns:
  Loss value for the traced batch.
)pbdoc")
      .def("fast_loop", &JITGraph::fast_loop, py::arg("dataloader"),
           py::call_guard<py::gil_scoped_release>(), R"pbdoc(
Replay the traced graph for the remaining batches in an epoch.

Args:
  dataloader: Batch source to consume.

Returns:
  Tuple of total loss and number of processed batches.
)pbdoc")
      .def("evaluate", &JITGraph::evaluate, py::arg("dataloader"),
           py::call_guard<py::gil_scoped_release>(), R"pbdoc(
Evaluate the traced graph without parameter updates.

Args:
  dataloader: Batch source to consume.

Returns:
  Average loss over the available batches.
)pbdoc")
      .def("fast_fit", &JITGraph::fast_fit, py::arg("dataloader"),
           py::arg("val_dataloader") = nullptr, py::arg("epochs"),
           py::arg("tol") = 1e-4f, py::arg("n_iter_no_change") = 10,
           py::arg("verbose") = true, py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
Train for multiple epochs with optional validation and early stopping.

Args:
  dataloader: Training batch source.
  val_dataloader: Optional validation batch source.
  epochs: Number of epochs to run.
  tol: Minimum improvement required to reset the early-stopping window.
  n_iter_no_change: Number of epochs without improvement before stop.
  verbose: Whether to print progress updates.

Notes:
  The traced training loop and validation pass execute natively in C++ with
  the GIL released.
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