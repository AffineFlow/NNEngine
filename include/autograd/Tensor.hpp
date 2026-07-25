#pragma once
#include <utility>

#include "core/Types.hpp"

namespace mlengine::autograd {

struct Tensor {
  mlengine::FlatStorage data;
  mlengine::FlatStorage grad;
  mlengine::Shape shape;
  bool requires_grad;
  bool apply_regularization;

  Tensor(const mlengine::Shape& init_shape, bool req_grad = true,
         bool apply_reg = false)
      : shape(init_shape),
        requires_grad(req_grad),
        apply_regularization(apply_reg) {
    Eigen::Index total_size = mlengine::compute_size(shape);
    data = mlengine::FlatStorage(total_size);
    data.setZero();

    if (requires_grad) {
      grad = mlengine::FlatStorage(total_size);
      grad.setZero();
    }
  }

  bool has_shape(const mlengine::Shape& target) const {
    return shape == target;
  }

  void resize(const mlengine::Shape& new_shape) {
    if (has_shape(new_shape)) return;
    shape = new_shape;
    Eigen::Index total_size = mlengine::compute_size(shape);
    data = mlengine::FlatStorage(total_size);
    if (requires_grad) {
      grad = mlengine::FlatStorage(total_size);
      grad.setZero();
    }
  }

  void zero_grad() {
    if (requires_grad) {
      grad.setZero();
    }
  }
};

}  // namespace mlengine::autograd