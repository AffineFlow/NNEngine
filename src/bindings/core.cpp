#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>

#include "autograd/Op.hpp"
#include "autograd/Tape.hpp"
#include "autograd/Tensor.hpp"
#include "autograd/ops/AddOp.hpp"
#include "autograd/ops/DivOp.hpp"
#include "autograd/ops/MatMulOp.hpp"
#include "autograd/ops/MulOp.hpp"
#include "autograd/ops/PyOp.hpp"
#include "autograd/ops/ReLUOp.hpp"
#include "autograd/ops/ReductionOps.hpp"
#include "autograd/ops/ScalarOps.hpp"
#include "autograd/ops/SubOp.hpp"
#include "autograd/ops/TransposeOp.hpp"
#include "autograd/ops/UnaryOps.hpp"
#include "core/DataLoader.hpp"
#include "core/Random.hpp"

namespace py = pybind11;
using namespace mlengine::core;

void bind_core_utils(py::module_& m) {
  m.def("set_seed", &set_seed, py::arg("seed"),
        "Seed the shared RNGs to make initialization, shuffling, and dropout "
        "repeatable.");

  py::class_<mlengine::autograd::Tensor>(
      m, "Tensor", py::buffer_protocol(),
      "A multi-dimensional array with autograd support.")
      .def(py::init(
               [](py::array_t<float, py::array::c_style | py::array::forcecast>
                      b) {
                 py::buffer_info info = b.request();
                 mlengine::Shape shape(info.ndim);
                 for (int i = 0; i < info.ndim; i++) shape[i] = info.shape[i];
                 mlengine::autograd::Tensor t(shape, false);
                 std::memcpy(t.data.data(), info.ptr,
                             info.size * sizeof(float));
                 return t;
               }),
           py::arg("data"), "Initialize a tensor from a NumPy array.")
      .def_buffer([](mlengine::autograd::Tensor& t) -> py::buffer_info {
        std::vector<py::ssize_t> py_shape(t.shape.begin(), t.shape.end());
        std::vector<py::ssize_t> strides(t.shape.size());
        py::ssize_t stride = sizeof(float);
        for (int i = t.shape.size() - 1; i >= 0; --i) {
          strides[i] = stride;
          stride *= t.shape[i];
        }
        return py::buffer_info(t.data.data(), sizeof(float),
                               py::format_descriptor<float>::format(),
                               t.shape.size(), py_shape, strides);
      })
      .def_property(
          "shape", [](const mlengine::autograd::Tensor& t) { return t.shape; },
          [](mlengine::autograd::Tensor& t, const mlengine::Shape& s) {
            t.resize(s);
          },
          "The dimensions of the tensor.")
      .def_property_readonly(
          "data",
          [](mlengine::autograd::Tensor& t) {
            std::vector<py::ssize_t> py_shape(t.shape.begin(), t.shape.end());
            std::vector<py::ssize_t> strides(t.shape.size());
            py::ssize_t stride = sizeof(float);
            for (int i = t.shape.size() - 1; i >= 0; --i) {
              strides[i] = stride;
              stride *= t.shape[i];
            }
            return py::array_t<float>(py_shape, strides, t.data.data(),
                                      py::cast(&t));
          },
          "Zero-copy NumPy view of the underlying forward data.")
      .def_property_readonly(
          "grad",
          [](mlengine::autograd::Tensor& t) {
            std::vector<py::ssize_t> py_shape(t.shape.begin(), t.shape.end());
            std::vector<py::ssize_t> strides(t.shape.size());
            py::ssize_t stride = sizeof(float);
            for (int i = t.shape.size() - 1; i >= 0; --i) {
              strides[i] = stride;
              stride *= t.shape[i];
            }
            return py::array_t<float>(py_shape, strides, t.grad.data(),
                                      py::cast(&t));
          },
          "Zero-copy NumPy view of the underlying gradient data.")
      .def_property(
          "requires_grad",
          [](const mlengine::autograd::Tensor& t) { return t.requires_grad; },
          [](mlengine::autograd::Tensor& t, bool value) {
            t.requires_grad = value;
          },
          "Whether this tensor should accumulate gradients during the backward "
          "pass.")
      .def("__repr__",
           [](const mlengine::autograd::Tensor& t) {
             std::ostringstream oss;
             oss << "Tensor(shape=(";
             for (size_t i = 0; i < t.shape.size(); ++i) {
               oss << t.shape[i] << (i == t.shape.size() - 1 ? "" : ", ");
             }
             oss << "), requires_grad=" << (t.requires_grad ? "True" : "False")
                 << ")";
             return oss.str();
           })
      .def(
          "__add__",
          [](mlengine::autograd::Tensor& self,
             mlengine::autograd::Tensor& other) {
            auto* tape = mlengine::autograd::Tape::get_global();
            bool req_grad = tape->record_ops_ &&
                            (self.requires_grad || other.requires_grad);
            auto* out = tape->alloc_tensor(self.shape, req_grad);
            auto op = std::make_shared<mlengine::autograd::ops::AddOp>(
                &self, &other, out);
            op->forward();
            tape->record_op(op);
            return out;
          },
          py::arg("other"),
          "Element-wise addition of two tensors. Records to the active "
          "autograd tape.",
          py::return_value_policy::reference)

      .def(
          "__sub__",
          [](mlengine::autograd::Tensor& self,
             mlengine::autograd::Tensor& other) {
            auto* tape = mlengine::autograd::Tape::get_global();
            bool req_grad = tape->record_ops_ &&
                            (self.requires_grad || other.requires_grad);
            auto* out = tape->alloc_tensor(self.shape, req_grad);
            auto op = std::make_shared<mlengine::autograd::ops::SubOp>(
                &self, &other, out);
            op->forward();
            tape->record_op(op);
            return out;
          },
          py::arg("other"),
          "Element-wise subtraction of two tensors. Records to the active "
          "autograd tape.",
          py::return_value_policy::reference)

      .def(
          "__mul__",
          [](mlengine::autograd::Tensor& self,
             mlengine::autograd::Tensor& other) {
            auto* tape = mlengine::autograd::Tape::get_global();
            bool req_grad = tape->record_ops_ &&
                            (self.requires_grad || other.requires_grad);
            auto* out = tape->alloc_tensor(self.shape, req_grad);
            auto op = std::make_shared<mlengine::autograd::ops::MulOp>(
                &self, &other, out);
            op->forward();
            tape->record_op(op);
            return out;
          },
          py::arg("other"),
          "Element-wise multiplication of two tensors. Records to the active "
          "autograd tape.",
          py::return_value_policy::reference)

      .def(
          "__truediv__",
          [](mlengine::autograd::Tensor& self,
             mlengine::autograd::Tensor& other) {
            auto* tape = mlengine::autograd::Tape::get_global();
            bool req_grad = tape->record_ops_ &&
                            (self.requires_grad || other.requires_grad);
            auto* out = tape->alloc_tensor(self.shape, req_grad);
            auto op = std::make_shared<mlengine::autograd::ops::DivOp>(
                &self, &other, out);
            op->forward();
            tape->record_op(op);
            return out;
          },
          py::arg("other"),
          "Element-wise division of two tensors. Records to the active "
          "autograd tape.",
          py::return_value_policy::reference)

      .def(
          "__matmul__",
          [](mlengine::autograd::Tensor& self,
             mlengine::autograd::Tensor& other) {
            auto* tape = mlengine::autograd::Tape::get_global();
            if (self.shape.size() != 2 || other.shape.size() != 2)
              throw std::invalid_argument("MatMul requires 2D tensors");
            bool req_grad = tape->record_ops_ &&
                            (self.requires_grad || other.requires_grad);
            auto* out =
                tape->alloc_tensor({self.shape[0], other.shape[1]}, req_grad);
            auto op = std::make_shared<mlengine::autograd::ops::MatMulOp>(
                &self, &other, out);
            op->forward();
            tape->record_op(op);
            return out;
          },
          py::arg("other"),
          "Matrix multiplication of two 2D tensors. Records to the active "
          "autograd tape.",
          py::return_value_policy::reference)
      .def(
          "sum",
          [](mlengine::autograd::Tensor& self) {
            auto* tape = mlengine::autograd::Tape::get_global();
            bool req_grad = tape->record_ops_ && self.requires_grad;
            auto* out = tape->alloc_tensor({1}, req_grad);
            auto op =
                std::make_shared<mlengine::autograd::ops::SumOp>(&self, out);
            op->forward();
            tape->record_op(op);
            return out;
          },
          "Sum of all elements. Records to the active autograd tape.",
          py::return_value_policy::reference)

      .def(
          "mean",
          [](mlengine::autograd::Tensor& self) {
            auto* tape = mlengine::autograd::Tape::get_global();
            bool req_grad = tape->record_ops_ && self.requires_grad;
            auto* out = tape->alloc_tensor({1}, req_grad);
            auto op =
                std::make_shared<mlengine::autograd::ops::MeanOp>(&self, out);
            op->forward();
            tape->record_op(op);
            return out;
          },
          "Mean of all elements. Records to the active autograd tape.",
          py::return_value_policy::reference)
      .def(
          "__add__",
          [](mlengine::autograd::Tensor& self, float val) {
            auto* tape = mlengine::autograd::Tape::get_global();
            bool req_grad = tape->record_ops_ && self.requires_grad;
            auto* out = tape->alloc_tensor(self.shape, req_grad);
            auto op = std::make_shared<mlengine::autograd::ops::AddScalarOp>(
                &self, val, out);
            op->forward();
            tape->record_op(op);
            return out;
          },
          py::arg("val"), py::return_value_policy::reference)
      .def(
          "__radd__",
          [](mlengine::autograd::Tensor& self, float val) {
            auto* tape = mlengine::autograd::Tape::get_global();
            bool req_grad = tape->record_ops_ && self.requires_grad;
            auto* out = tape->alloc_tensor(self.shape, req_grad);
            auto op = std::make_shared<mlengine::autograd::ops::AddScalarOp>(
                &self, val, out);  // Addition is commutative
            op->forward();
            tape->record_op(op);
            return out;
          },
          py::arg("val"), py::return_value_policy::reference)

      .def(
          "__sub__",
          [](mlengine::autograd::Tensor& self, float val) {
            auto* tape = mlengine::autograd::Tape::get_global();
            bool req_grad = tape->record_ops_ && self.requires_grad;
            auto* out = tape->alloc_tensor(self.shape, req_grad);
            auto op = std::make_shared<mlengine::autograd::ops::SubScalarOp>(
                &self, val, out);
            op->forward();
            tape->record_op(op);
            return out;
          },
          py::arg("val"), py::return_value_policy::reference)
      .def(
          "__rsub__",
          [](mlengine::autograd::Tensor& self, float val) {
            auto* tape = mlengine::autograd::Tape::get_global();
            bool req_grad = tape->record_ops_ && self.requires_grad;
            auto* out = tape->alloc_tensor(self.shape, req_grad);
            auto op = std::make_shared<mlengine::autograd::ops::RSubScalarOp>(
                &self, val, out);
            op->forward();
            tape->record_op(op);
            return out;
          },
          py::arg("val"), py::return_value_policy::reference)

      .def(
          "__mul__",
          [](mlengine::autograd::Tensor& self, float val) {
            auto* tape = mlengine::autograd::Tape::get_global();
            bool req_grad = tape->record_ops_ && self.requires_grad;
            auto* out = tape->alloc_tensor(self.shape, req_grad);
            auto op = std::make_shared<mlengine::autograd::ops::MulScalarOp>(
                &self, val, out);
            op->forward();
            tape->record_op(op);
            return out;
          },
          py::arg("val"), py::return_value_policy::reference)
      .def(
          "__rmul__",
          [](mlengine::autograd::Tensor& self, float val) {
            auto* tape = mlengine::autograd::Tape::get_global();
            bool req_grad = tape->record_ops_ && self.requires_grad;
            auto* out = tape->alloc_tensor(self.shape, req_grad);
            auto op = std::make_shared<mlengine::autograd::ops::MulScalarOp>(
                &self, val, out);  // Multiplication is commutative
            op->forward();
            tape->record_op(op);
            return out;
          },
          py::arg("val"), py::return_value_policy::reference)

      .def(
          "__truediv__",
          [](mlengine::autograd::Tensor& self, float val) {
            auto* tape = mlengine::autograd::Tape::get_global();
            bool req_grad = tape->record_ops_ && self.requires_grad;
            auto* out = tape->alloc_tensor(self.shape, req_grad);
            auto op = std::make_shared<mlengine::autograd::ops::DivScalarOp>(
                &self, val, out);
            op->forward();
            tape->record_op(op);
            return out;
          },
          py::arg("val"), py::return_value_policy::reference)
      .def(
          "__rtruediv__",
          [](mlengine::autograd::Tensor& self, float val) {
            auto* tape = mlengine::autograd::Tape::get_global();
            bool req_grad = tape->record_ops_ && self.requires_grad;
            auto* out = tape->alloc_tensor(self.shape, req_grad);
            auto op = std::make_shared<mlengine::autograd::ops::RDivScalarOp>(
                &self, val, out);
            op->forward();
            tape->record_op(op);
            return out;
          },
          py::arg("val"), py::return_value_policy::reference)
      .def(
          "__iadd__",
          [](mlengine::autograd::Tensor& self,
             mlengine::autograd::Tensor& other) {
            self.data += other.data;
            return self;
          },
          py::arg("other"), "In-place addition (detached from autograd).",
          py::return_value_policy::reference)
      .def(
          "__iadd__",
          [](mlengine::autograd::Tensor& self, float val) {
            self.data = self.data + val;
            return self;
          },
          py::arg("val"), py::return_value_policy::reference)

      .def(
          "__isub__",
          [](mlengine::autograd::Tensor& self,
             mlengine::autograd::Tensor& other) {
            self.data -= other.data;
            return self;
          },
          py::arg("other"), "In-place subtraction (detached from autograd).",
          py::return_value_policy::reference)
      .def(
          "__isub__",
          [](mlengine::autograd::Tensor& self, float val) {
            self.data = self.data - val;
            return self;
          },
          py::arg("val"), py::return_value_policy::reference)

      .def(
          "__imul__",
          [](mlengine::autograd::Tensor& self,
             mlengine::autograd::Tensor& other) {
            self.data *= other.data;
            return self;
          },
          py::arg("other"), "In-place multiplication (detached from autograd).",
          py::return_value_policy::reference)
      .def(
          "__imul__",
          [](mlengine::autograd::Tensor& self, float val) {
            self.data = self.data * val;
            return self;
          },
          py::arg("val"), py::return_value_policy::reference)
      .def(
          "backward",
          [](mlengine::autograd::Tensor& self) {
            auto* tape = mlengine::autograd::Tape::get_global();
            if (!tape)
              throw std::runtime_error(
                  "Cannot call backward() outside of a Tape context.");
            self.grad.setConstant(1.0f);
            tape->backward();
          },
          "Computes the gradient of current tensor w.r.t. graph leaves.")
      .def(
          "exp",
          [](mlengine::autograd::Tensor& self) {
            auto* tape = mlengine::autograd::Tape::get_global();
            bool req_grad = tape->record_ops_ && self.requires_grad;
            auto* out = tape->alloc_tensor(self.shape, req_grad);

            auto op =
                std::make_shared<mlengine::autograd::ops::ExpOp>(&self, out);
            op->forward();
            tape->record_op(op);

            return out;
          },
          "Element-wise exponential.", py::return_value_policy::reference)

      .def(
          "log",
          [](mlengine::autograd::Tensor& self) {
            auto* tape = mlengine::autograd::Tape::get_global();
            bool req_grad = tape->record_ops_ && self.requires_grad;
            auto* out = tape->alloc_tensor(self.shape, req_grad);

            auto op =
                std::make_shared<mlengine::autograd::ops::LogOp>(&self, out);
            op->forward();
            tape->record_op(op);

            return out;
          },
          "Element-wise natural logarithm.", py::return_value_policy::reference)
      .def_property_readonly(
          "T",
          [](mlengine::autograd::Tensor& self) {
            if (self.shape.size() != 2) {
              throw std::runtime_error(
                  "T (transpose) currently only supports 2D tensors.");
            }

            auto* tape = mlengine::autograd::Tape::get_global();
            bool req_grad = tape->record_ops_ && self.requires_grad;
            auto* out =
                tape->alloc_tensor({self.shape[1], self.shape[0]}, req_grad);

            auto op = std::make_shared<mlengine::autograd::ops::TransposeOp>(
                &self, out);
            op->forward();
            tape->record_op(op);

            return out;
          },
          "Returns a view of the 2D tensor with its dimensions reversed.",
          py::return_value_policy::reference)
      .def(
          "transpose",
          [](mlengine::autograd::Tensor& self) {
            if (self.shape.size() != 2) {
              throw std::runtime_error(
                  "transpose() currently only supports 2D tensors.");
            }
            auto* tape = mlengine::autograd::Tape::get_global();
            bool req_grad = tape->record_ops_ && self.requires_grad;
            auto* out =
                tape->alloc_tensor({self.shape[1], self.shape[0]}, req_grad);
            auto op = std::make_shared<mlengine::autograd::ops::TransposeOp>(
                &self, out);
            op->forward();
            tape->record_op(op);
            return out;
          },
          "Transposes a 2D tensor.", py::return_value_policy::reference)
      .def(
          "relu",
          [](mlengine::autograd::Tensor& self) {
            auto* tape = mlengine::autograd::Tape::get_global();
            bool req_grad = tape->record_ops_ && self.requires_grad;
            auto* out = tape->alloc_tensor(self.shape, req_grad);

            auto op =
                std::make_shared<mlengine::autograd::ops::ReLUOp>(&self, out);
            op->forward();
            tape->record_op(op);

            return out;
          },
          "Element-wise rectified linear activation.",
          py::return_value_policy::reference);

  py::implicitly_convertible<py::array_t<float>, mlengine::autograd::Tensor>();

  py::class_<mlengine::autograd::Op, mlengine::autograd::ops::PyOp,
             std::shared_ptr<mlengine::autograd::Op>>(
      m, "Op", "Base interface for a differentiable primitive.")
      .def(py::init<>())
      .def("forward", &mlengine::autograd::Op::forward,
           "Execute the forward pass for the primitive.")
      .def("backward", &mlengine::autograd::Op::backward,
           "Accumulate gradients for the primitive inputs.");

  py::class_<mlengine::autograd::Tape,
             std::shared_ptr<mlengine::autograd::Tape>>(
      m, "Tape", "Context manager for recording operations for autograd.")
      .def(py::init<bool>(), py::arg("record_ops") = true)
      .def("__enter__",
           [](mlengine::autograd::Tape& t) {
             mlengine::autograd::Tape::set_global(&t);
             return &t;
           })
      .def("__exit__",
           [](mlengine::autograd::Tape& t, py::object exc_type,
              py::object exc_value, py::object traceback) {
             mlengine::autograd::Tape::set_global(nullptr);
           })
      .def_property(
          "record_ops",
          [](const mlengine::autograd::Tape& t) { return t.record_ops_; },
          [](mlengine::autograd::Tape& t, bool value) {
            t.record_ops_ = value;
          },
          "Whether the tape is actively recording operations.")
      .def("alloc_tensor", &mlengine::autograd::Tape::alloc_tensor,
           py::arg("shape"), py::arg("requires_grad") = true,
           "Allocate a zeroed tensor from the tape's memory pool.",
           py::return_value_policy::reference)
      .def("push_tensor", &mlengine::autograd::Tape::push_tensor,
           py::arg("data"), py::arg("requires_grad") = true,
           "Push an existing tensor onto the tape, copying its data.",
           py::return_value_policy::reference)
      .def("record_op", &mlengine::autograd::Tape::record_op, py::arg("op"),
           "Register a primitive operation on the tape.")
      .def("backward", &mlengine::autograd::Tape::backward,
           "Execute the reverse-mode accumulation (backpropagation).")
      .def("reset", &mlengine::autograd::Tape::reset,
           "Clear the tape's recorded ops and reset the memory pool index.");

  py::class_<DataLoader>(
      m, "DataLoader", "Batch generator for training and evaluation datasets.")
      .def(py::init<const mlengine::autograd::Tensor&,
                    const mlengine::autograd::Tensor&, size_t, bool, bool>(),
           py::arg("X"), py::arg("y"), py::arg("batch_size"),
           py::arg("shuffle") = true, py::arg("drop_last") = false,
           "Initialize a data loader with features and targets.")
      .def("reset", &DataLoader::reset,
           "Reset the loader to the beginning of the dataset and optionally "
           "reshuffle.")
      .def("has_next", &DataLoader::has_next,
           "Check if there are remaining batches in the current epoch.")
      .def("next_batch", &DataLoader::next_batch, py::arg("X_batch"),
           py::arg("y_batch"),
           "Populate the provided tensors with the next batch of data.");
}