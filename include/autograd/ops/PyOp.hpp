#pragma once

#include <pybind11/pybind11.h>

#include "autograd/Op.hpp"

namespace affineengine::autograd::ops {

/**
 * @brief Python trampoline that forwards virtual calls into Python overrides.
 *
 * Python users can inherit from Op to define custom forward/backward passes
 * while still participating in the native tape replay loop.
 */
class PyOp : public affineengine::autograd::Op {
 public:
  using affineengine::autograd::Op::Op;

  void forward() override {
    PYBIND11_OVERRIDE_PURE(void, affineengine::autograd::Op, forward);
  }

  void backward() override {
    PYBIND11_OVERRIDE_PURE(void, affineengine::autograd::Op, backward);
  }
};

}  // namespace affineengine::autograd::ops