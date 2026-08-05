#pragma once

#include <pybind11/pybind11.h>

#include "autograd/Op.hpp"

namespace affineflow::nn::autograd::ops {

/**
 * @brief Python trampoline that forwards virtual calls into Python overrides.
 *
 * Python users can inherit from Op to define custom forward/backward passes
 * while still participating in the native tape replay loop.
 */
class PyOp : public affineflow::nn::autograd::Op {
 public:
  using affineflow::nn::autograd::Op::Op;

  void forward() override {
    PYBIND11_OVERRIDE_PURE(void, affineflow::nn::autograd::Op, forward);
  }

  void backward() override {
    PYBIND11_OVERRIDE_PURE(void, affineflow::nn::autograd::Op, backward);
  }
};

}  // namespace affineflow::nn::autograd::ops