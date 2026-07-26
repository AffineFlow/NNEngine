#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>

#include "autograd/Op.hpp"
#include "autograd/Tape.hpp"
#include "autograd/Tensor.hpp"
#include "autograd/ops/PyOp.hpp"
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
      .def("__repr__", [](const mlengine::autograd::Tensor& t) {
        std::ostringstream oss;
        oss << "Tensor(shape=(";
        for (size_t i = 0; i < t.shape.size(); ++i) {
          oss << t.shape[i] << (i == t.shape.size() - 1 ? "" : ", ");
        }
        oss << "), requires_grad=" << (t.requires_grad ? "True" : "False")
            << ")";
        return oss.str();
      });

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