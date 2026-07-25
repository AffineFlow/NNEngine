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
  m.def("set_seed", &set_seed, py::arg("seed"));

  py::class_<mlengine::autograd::Tensor>(m, "Tensor", py::buffer_protocol())
      .def(py::init(
          [](py::array_t<float, py::array::c_style | py::array::forcecast> b) {
            py::buffer_info info = b.request();
            mlengine::Shape shape(info.ndim);
            for (int i = 0; i < info.ndim; i++) shape[i] = info.shape[i];
            mlengine::autograd::Tensor t(shape, false);
            std::memcpy(t.data.data(), info.ptr, info.size * sizeof(float));
            return t;
          }))
      // Native Buffer Protocol: Allows np.array(tensor) directly with correct
      // shape
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
          })
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
          })
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
          })
      .def_property(
          "requires_grad",
          [](const mlengine::autograd::Tensor& t) { return t.requires_grad; },
          [](mlengine::autograd::Tensor& t, bool value) {
            t.requires_grad = value;
          })
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
           py::arg("shape"), py::arg("requires_grad") = true,
           py::return_value_policy::reference)
      .def("push_tensor", &mlengine::autograd::Tape::push_tensor,
           py::arg("data"), py::arg("requires_grad") = true,
           py::return_value_policy::reference)
      .def("record_op", &mlengine::autograd::Tape::record_op, py::arg("op"))
      .def("backward", &mlengine::autograd::Tape::backward)
      .def("reset", &mlengine::autograd::Tape::reset);

  py::class_<DataLoader>(m, "DataLoader")
      .def(py::init<const mlengine::autograd::Tensor&,
                    const mlengine::autograd::Tensor&, size_t, bool, bool>(),
           py::arg("X"), py::arg("y"), py::arg("batch_size"),
           py::arg("shuffle") = true, py::arg("drop_last") = false)
      .def("reset", &DataLoader::reset)
      .def("has_next", &DataLoader::has_next)
      .def("next_batch", &DataLoader::next_batch, py::arg("X_batch"),
           py::arg("y_batch"));
}