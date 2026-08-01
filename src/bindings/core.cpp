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
using namespace affineengine::core;

void bind_core_utils(py::module_& m) {
  m.def("set_seed", &set_seed, py::arg("seed"),
        "Seed the shared RNGs to make initialization, shuffling, and dropout "
        "repeatable.");

  py::class_<affineengine::autograd::Tensor>(
      m, "Tensor", py::buffer_protocol(),
      "A multi-dimensional array with autograd support.")
      .def(py::init(
               [](py::array_t<float, py::array::c_style | py::array::forcecast>
                      b) {
                 py::buffer_info info = b.request();
                 affineengine::Shape shape(info.ndim);
                 for (int i = 0; i < info.ndim; i++) shape[i] = info.shape[i];
                 affineengine::autograd::Tensor t(shape, false);
                 std::memcpy(t.data.data(), info.ptr,
                             info.size * sizeof(float));
                 return t;
               }),
           py::arg("data"), "Initialize a tensor from a NumPy array.")
      .def_buffer([](affineengine::autograd::Tensor& t) -> py::buffer_info {
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
          "shape", [](const affineengine::autograd::Tensor& t) { return t.shape; },
          [](affineengine::autograd::Tensor& t, const affineengine::Shape& s) {
            t.resize(s);
          },
          "The dimensions of the tensor.")
      .def_property_readonly(
          "data",
          [](affineengine::autograd::Tensor& t) {
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
          [](affineengine::autograd::Tensor& t) {
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
          [](const affineengine::autograd::Tensor& t) { return t.requires_grad; },
          [](affineengine::autograd::Tensor& t, bool value) {
            t.requires_grad = value;
          },
          "Whether this tensor should accumulate gradients during the backward "
          "pass.")
      .def("__repr__",
           [](const affineengine::autograd::Tensor& t) {
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
          [](affineengine::autograd::Tensor& self,
             affineengine::autograd::Tensor& other) -> py::object {
            auto* tape = affineengine::autograd::Tape::global_tape();
            if (!tape) {
              affineengine::autograd::Tensor out(self.shape, false);
              out.data = self.data + other.data;
              return py::cast(out);
            }
            bool req_grad = tape->record_ops_ &&
                            (self.requires_grad || other.requires_grad);
            auto* out = tape->alloc_tensor(self.shape, req_grad);
            auto* op = tape->allocate_op<affineengine::autograd::ops::AddOp>(
                &self, &other, out);
            op->forward();
            return py::cast(out, py::return_value_policy::reference);
          },
          py::arg("other"), "Element-wise addition of two tensors.")
      .def(
          "__sub__",
          [](affineengine::autograd::Tensor& self,
             affineengine::autograd::Tensor& other) -> py::object {
            auto* tape = affineengine::autograd::Tape::global_tape();
            if (!tape) {
              affineengine::autograd::Tensor out(self.shape, false);
              out.data = self.data - other.data;
              return py::cast(out);
            }
            bool req_grad = tape->record_ops_ &&
                            (self.requires_grad || other.requires_grad);
            auto* out = tape->alloc_tensor(self.shape, req_grad);
            auto* op = tape->allocate_op<affineengine::autograd::ops::SubOp>(
                &self, &other, out);
            op->forward();
            return py::cast(out, py::return_value_policy::reference);
          },
          py::arg("other"), "Element-wise subtraction of two tensors.")
      .def(
          "__mul__",
          [](affineengine::autograd::Tensor& self,
             affineengine::autograd::Tensor& other) -> py::object {
            auto* tape = affineengine::autograd::Tape::global_tape();
            if (!tape) {
              affineengine::autograd::Tensor out(self.shape, false);
              out.data = self.data * other.data;
              return py::cast(out);
            }
            bool req_grad = tape->record_ops_ &&
                            (self.requires_grad || other.requires_grad);
            auto* out = tape->alloc_tensor(self.shape, req_grad);
            auto* op = tape->allocate_op<affineengine::autograd::ops::MulOp>(
                &self, &other, out);
            op->forward();
            return py::cast(out, py::return_value_policy::reference);
          },
          py::arg("other"), "Element-wise multiplication of two tensors.")
      .def(
          "__truediv__",
          [](affineengine::autograd::Tensor& self,
             affineengine::autograd::Tensor& other) -> py::object {
            auto* tape = affineengine::autograd::Tape::global_tape();
            if (!tape) {
              affineengine::autograd::Tensor out(self.shape, false);
              out.data = self.data / other.data;
              return py::cast(out);
            }
            bool req_grad = tape->record_ops_ &&
                            (self.requires_grad || other.requires_grad);
            auto* out = tape->alloc_tensor(self.shape, req_grad);
            auto* op = tape->allocate_op<affineengine::autograd::ops::DivOp>(
                &self, &other, out);
            op->forward();
            return py::cast(out, py::return_value_policy::reference);
          },
          py::arg("other"), "Element-wise division of two tensors.")
      .def(
          "__matmul__",
          [](affineengine::autograd::Tensor& self,
             affineengine::autograd::Tensor& other) -> py::object {
            if (self.shape.size() != 2 || other.shape.size() != 2)
              throw std::invalid_argument("MatMul requires 2D tensors");
            auto* tape = affineengine::autograd::Tape::global_tape();
            if (!tape) {
              affineengine::autograd::Tensor out({self.shape[0], other.shape[1]},
                                             false);
              Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::RowMajor>>
                  A_mat(self.data.data(), self.shape[0], self.shape[1]);
              Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::RowMajor>>
                  B_mat(other.data.data(), other.shape[0], other.shape[1]);
              Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::RowMajor>>
                  Out_mat(out.data.data(), out.shape[0], out.shape[1]);
              Out_mat.noalias() = A_mat * B_mat;
              return py::cast(out);
            }
            bool req_grad = tape->record_ops_ &&
                            (self.requires_grad || other.requires_grad);
            auto* out =
                tape->alloc_tensor({self.shape[0], other.shape[1]}, req_grad);
            auto* op = tape->allocate_op<affineengine::autograd::ops::MatMulOp>(
                &self, &other, out);
            op->forward();
            return py::cast(out, py::return_value_policy::reference);
          },
          py::arg("other"), "Matrix multiplication of two 2D tensors.")
      .def(
          "sum",
          [](affineengine::autograd::Tensor& self) -> py::object {
            auto* tape = affineengine::autograd::Tape::global_tape();
            if (!tape) {
              affineengine::autograd::Tensor out({1}, false);
              Eigen::Map<Eigen::ArrayXf> a_arr(self.data.data(),
                                               self.data.size());
              out.data.data()[0] = a_arr.sum();
              return py::cast(out);
            }
            bool req_grad = tape->record_ops_ && self.requires_grad;
            auto* out = tape->alloc_tensor({1}, req_grad);
            auto* op =
                tape->allocate_op<affineengine::autograd::ops::SumOp>(&self, out);
            op->forward();
            return py::cast(out, py::return_value_policy::reference);
          },
          "Sum of all elements.")
      .def(
          "mean",
          [](affineengine::autograd::Tensor& self) -> py::object {
            auto* tape = affineengine::autograd::Tape::global_tape();
            if (!tape) {
              affineengine::autograd::Tensor out({1}, false);
              Eigen::Map<Eigen::ArrayXf> a_arr(self.data.data(),
                                               self.data.size());
              out.data.data()[0] = a_arr.mean();
              return py::cast(out);
            }
            bool req_grad = tape->record_ops_ && self.requires_grad;
            auto* out = tape->alloc_tensor({1}, req_grad);
            auto* op =
                tape->allocate_op<affineengine::autograd::ops::MeanOp>(&self, out);
            op->forward();
            return py::cast(out, py::return_value_policy::reference);
          },
          "Mean of all elements.")
      .def(
          "__add__",
          [](affineengine::autograd::Tensor& self, float val) -> py::object {
            auto* tape = affineengine::autograd::Tape::global_tape();
            if (!tape) {
              affineengine::autograd::Tensor out(self.shape, false);
              out.data = self.data + val;
              return py::cast(out);
            }
            bool req_grad = tape->record_ops_ && self.requires_grad;
            auto* out = tape->alloc_tensor(self.shape, req_grad);
            auto* op = tape->allocate_op<affineengine::autograd::ops::AddScalarOp>(
                &self, val, out);
            op->forward();
            return py::cast(out, py::return_value_policy::reference);
          },
          py::arg("val"))
      .def(
          "__radd__",
          [](affineengine::autograd::Tensor& self, float val) -> py::object {
            auto* tape = affineengine::autograd::Tape::global_tape();
            if (!tape) {
              affineengine::autograd::Tensor out(self.shape, false);
              out.data = self.data + val;
              return py::cast(out);
            }
            bool req_grad = tape->record_ops_ && self.requires_grad;
            auto* out = tape->alloc_tensor(self.shape, req_grad);
            auto* op = tape->allocate_op<affineengine::autograd::ops::AddScalarOp>(
                &self, val, out);
            op->forward();
            return py::cast(out, py::return_value_policy::reference);
          },
          py::arg("val"))
      .def(
          "__sub__",
          [](affineengine::autograd::Tensor& self, float val) -> py::object {
            auto* tape = affineengine::autograd::Tape::global_tape();
            if (!tape) {
              affineengine::autograd::Tensor out(self.shape, false);
              out.data = self.data - val;
              return py::cast(out);
            }
            bool req_grad = tape->record_ops_ && self.requires_grad;
            auto* out = tape->alloc_tensor(self.shape, req_grad);
            auto* op = tape->allocate_op<affineengine::autograd::ops::SubScalarOp>(
                &self, val, out);
            op->forward();
            return py::cast(out, py::return_value_policy::reference);
          },
          py::arg("val"))
      .def(
          "__rsub__",
          [](affineengine::autograd::Tensor& self, float val) -> py::object {
            auto* tape = affineengine::autograd::Tape::global_tape();
            if (!tape) {
              affineengine::autograd::Tensor out(self.shape, false);
              out.data = val - self.data;
              return py::cast(out);
            }
            bool req_grad = tape->record_ops_ && self.requires_grad;
            auto* out = tape->alloc_tensor(self.shape, req_grad);
            auto* op = tape->allocate_op<affineengine::autograd::ops::RSubScalarOp>(
                &self, val, out);
            op->forward();
            return py::cast(out, py::return_value_policy::reference);
          },
          py::arg("val"))
      .def(
          "__mul__",
          [](affineengine::autograd::Tensor& self, float val) -> py::object {
            auto* tape = affineengine::autograd::Tape::global_tape();
            if (!tape) {
              affineengine::autograd::Tensor out(self.shape, false);
              out.data = self.data * val;
              return py::cast(out);
            }
            bool req_grad = tape->record_ops_ && self.requires_grad;
            auto* out = tape->alloc_tensor(self.shape, req_grad);
            auto* op = tape->allocate_op<affineengine::autograd::ops::MulScalarOp>(
                &self, val, out);
            op->forward();
            return py::cast(out, py::return_value_policy::reference);
          },
          py::arg("val"))
      .def(
          "__rmul__",
          [](affineengine::autograd::Tensor& self, float val) -> py::object {
            auto* tape = affineengine::autograd::Tape::global_tape();
            if (!tape) {
              affineengine::autograd::Tensor out(self.shape, false);
              out.data = self.data * val;
              return py::cast(out);
            }
            bool req_grad = tape->record_ops_ && self.requires_grad;
            auto* out = tape->alloc_tensor(self.shape, req_grad);
            auto* op = tape->allocate_op<affineengine::autograd::ops::MulScalarOp>(
                &self, val, out);
            op->forward();
            return py::cast(out, py::return_value_policy::reference);
          },
          py::arg("val"))
      .def(
          "__truediv__",
          [](affineengine::autograd::Tensor& self, float val) -> py::object {
            auto* tape = affineengine::autograd::Tape::global_tape();
            if (!tape) {
              affineengine::autograd::Tensor out(self.shape, false);
              out.data = self.data / val;
              return py::cast(out);
            }
            bool req_grad = tape->record_ops_ && self.requires_grad;
            auto* out = tape->alloc_tensor(self.shape, req_grad);
            auto* op = tape->allocate_op<affineengine::autograd::ops::DivScalarOp>(
                &self, val, out);
            op->forward();
            return py::cast(out, py::return_value_policy::reference);
          },
          py::arg("val"))
      .def(
          "__rtruediv__",
          [](affineengine::autograd::Tensor& self, float val) -> py::object {
            auto* tape = affineengine::autograd::Tape::global_tape();
            if (!tape) {
              affineengine::autograd::Tensor out(self.shape, false);
              out.data = val / self.data;
              return py::cast(out);
            }
            bool req_grad = tape->record_ops_ && self.requires_grad;
            auto* out = tape->alloc_tensor(self.shape, req_grad);
            auto* op = tape->allocate_op<affineengine::autograd::ops::RDivScalarOp>(
                &self, val, out);
            op->forward();
            return py::cast(out, py::return_value_policy::reference);
          },
          py::arg("val"))
      .def(
          "__iadd__",
          [](affineengine::autograd::Tensor& self,
             affineengine::autograd::Tensor& other) -> affineengine::autograd::Tensor& {
            self.data += other.data;
            return self;
          },
          py::arg("other"), "In-place addition (detached from autograd).",
          py::return_value_policy::reference)
      .def(
          "__iadd__",
          [](affineengine::autograd::Tensor& self,
             float val) -> affineengine::autograd::Tensor& {
            self.data = self.data + val;
            return self;
          },
          py::arg("val"), py::return_value_policy::reference)
      .def(
          "__isub__",
          [](affineengine::autograd::Tensor& self,
             affineengine::autograd::Tensor& other) -> affineengine::autograd::Tensor& {
            self.data -= other.data;
            return self;
          },
          py::arg("other"), "In-place subtraction (detached from autograd).",
          py::return_value_policy::reference)
      .def(
          "__isub__",
          [](affineengine::autograd::Tensor& self,
             float val) -> affineengine::autograd::Tensor& {
            self.data = self.data - val;
            return self;
          },
          py::arg("val"), py::return_value_policy::reference)
      .def(
          "__imul__",
          [](affineengine::autograd::Tensor& self,
             affineengine::autograd::Tensor& other) -> affineengine::autograd::Tensor& {
            self.data *= other.data;
            return self;
          },
          py::arg("other"), "In-place multiplication (detached from autograd).",
          py::return_value_policy::reference)
      .def(
          "__imul__",
          [](affineengine::autograd::Tensor& self,
             float val) -> affineengine::autograd::Tensor& {
            self.data = self.data * val;
            return self;
          },
          py::arg("val"), py::return_value_policy::reference)
      .def(
          "backward",
          [](affineengine::autograd::Tensor& self) {
            auto* tape = affineengine::autograd::Tape::get_global();
            if (!tape)
              throw std::runtime_error(
                  "Cannot call backward() outside of a Tape context.");
            self.grad.setConstant(1.0f);
            tape->backward();
          },
          "Computes the gradient of current tensor w.r.t. graph leaves.",
          py::call_guard<py::gil_scoped_release>())
      .def(
          "exp",
          [](affineengine::autograd::Tensor& self) -> py::object {
            auto* tape = affineengine::autograd::Tape::global_tape();
            if (!tape) {
              affineengine::autograd::Tensor out(self.shape, false);
              out.data = self.data.exp();
              return py::cast(out);
            }
            bool req_grad = tape->record_ops_ && self.requires_grad;
            auto* out = tape->alloc_tensor(self.shape, req_grad);
            auto* op =
                tape->allocate_op<affineengine::autograd::ops::ExpOp>(&self, out);
            op->forward();
            return py::cast(out, py::return_value_policy::reference);
          },
          "Element-wise exponential.")
      .def(
          "log",
          [](affineengine::autograd::Tensor& self) -> py::object {
            auto* tape = affineengine::autograd::Tape::global_tape();
            if (!tape) {
              affineengine::autograd::Tensor out(self.shape, false);
              out.data = self.data.log();
              return py::cast(out);
            }
            bool req_grad = tape->record_ops_ && self.requires_grad;
            auto* out = tape->alloc_tensor(self.shape, req_grad);
            auto* op =
                tape->allocate_op<affineengine::autograd::ops::LogOp>(&self, out);
            op->forward();
            return py::cast(out, py::return_value_policy::reference);
          },
          "Element-wise natural logarithm.")
      .def_property_readonly(
          "T",
          [](affineengine::autograd::Tensor& self) -> py::object {
            if (self.shape.size() != 2) {
              throw std::runtime_error(
                  "T (transpose) currently only supports 2D tensors.");
            }
            auto* tape = affineengine::autograd::Tape::global_tape();
            if (!tape) {
              affineengine::autograd::Tensor out({self.shape[1], self.shape[0]},
                                             false);
              Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::RowMajor>>
                  a_mat(self.data.data(), self.shape[0], self.shape[1]);
              Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::RowMajor>>
                  out_mat(out.data.data(), out.shape[0], out.shape[1]);
              out_mat = a_mat.transpose();
              return py::cast(out);
            }
            bool req_grad = tape->record_ops_ && self.requires_grad;
            auto* out =
                tape->alloc_tensor({self.shape[1], self.shape[0]}, req_grad);
            auto* op = tape->allocate_op<affineengine::autograd::ops::TransposeOp>(
                &self, out);
            op->forward();
            return py::cast(out, py::return_value_policy::reference);
          },
          "Returns a view of the 2D tensor with its dimensions reversed.")
      .def(
          "transpose",
          [](affineengine::autograd::Tensor& self) -> py::object {
            if (self.shape.size() != 2) {
              throw std::runtime_error(
                  "transpose() currently only supports 2D tensors.");
            }
            auto* tape = affineengine::autograd::Tape::global_tape();
            if (!tape) {
              affineengine::autograd::Tensor out({self.shape[1], self.shape[0]},
                                             false);
              Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::RowMajor>>
                  a_mat(self.data.data(), self.shape[0], self.shape[1]);
              Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                                       Eigen::RowMajor>>
                  out_mat(out.data.data(), out.shape[0], out.shape[1]);
              out_mat = a_mat.transpose();
              return py::cast(out);
            }
            bool req_grad = tape->record_ops_ && self.requires_grad;
            auto* out =
                tape->alloc_tensor({self.shape[1], self.shape[0]}, req_grad);
            auto* op = tape->allocate_op<affineengine::autograd::ops::TransposeOp>(
                &self, out);
            op->forward();
            return py::cast(out, py::return_value_policy::reference);
          },
          "Transposes a 2D tensor.")
      .def(
          "relu",
          [](affineengine::autograd::Tensor& self) -> py::object {
            auto* tape = affineengine::autograd::Tape::global_tape();
            if (!tape) {
              affineengine::autograd::Tensor out(self.shape, false);
              out.data = self.data.cwiseMax(0.0f);
              return py::cast(out);
            }
            bool req_grad = tape->record_ops_ && self.requires_grad;
            auto* out = tape->alloc_tensor(self.shape, req_grad);
            auto* op =
                tape->allocate_op<affineengine::autograd::ops::ReLUOp>(&self, out);
            op->forward();
            return py::cast(out, py::return_value_policy::reference);
          },
          "Element-wise rectified linear activation.");

  py::implicitly_convertible<py::array_t<float>, affineengine::autograd::Tensor>();

  py::class_<affineengine::autograd::Op, affineengine::autograd::ops::PyOp,
             std::shared_ptr<affineengine::autograd::Op>>(
      m, "Op", "Base interface for a differentiable primitive.")
      .def(py::init<>())
      .def("forward", &affineengine::autograd::Op::forward,
           "Execute the forward pass for the primitive.")
      .def("backward", &affineengine::autograd::Op::backward,
           "Accumulate gradients for the primitive inputs.");

  py::class_<affineengine::autograd::Tape,
             std::shared_ptr<affineengine::autograd::Tape>>(
      m, "Tape", "Context manager for recording operations for autograd.")
      .def(py::init<bool>(), py::arg("record_ops") = true)
      .def("__enter__",
           [](affineengine::autograd::Tape& t) {
             affineengine::autograd::Tape::set_global(&t);
             return &t;
           })
      .def("__exit__",
           [](affineengine::autograd::Tape& t, py::object exc_type,
              py::object exc_value, py::object traceback) {
             affineengine::autograd::Tape::set_global(nullptr);
           })
      .def_property(
          "record_ops",
          [](const affineengine::autograd::Tape& t) { return t.record_ops_; },
          [](affineengine::autograd::Tape& t, bool value) {
            t.record_ops_ = value;
          },
          "Whether the tape is actively recording operations.")
      .def("alloc_tensor", &affineengine::autograd::Tape::alloc_tensor,
           py::arg("shape"), py::arg("requires_grad") = true,
           "Allocate a zeroed tensor from the tape's memory pool.",
           py::return_value_policy::reference)
      .def("push_tensor", &affineengine::autograd::Tape::push_tensor,
           py::arg("data"), py::arg("requires_grad") = true,
           "Push an existing tensor onto the tape, copying its data.",
           py::return_value_policy::reference)
      .def("backward", &affineengine::autograd::Tape::backward,
           "Execute the reverse-mode accumulation (backpropagation).",
           py::call_guard<py::gil_scoped_release>())
      .def("reset", &affineengine::autograd::Tape::reset,
           "Clear the tape's recorded ops and reset the memory pool index.");

  py::class_<DataLoader>(
      m, "DataLoader", "Batch generator for training and evaluation datasets.")
      .def(py::init<const affineengine::autograd::Tensor&,
                    const affineengine::autograd::Tensor&, size_t, bool, bool>(),
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