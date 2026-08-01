#pragma once
#include <utility>

#include "core/Types.hpp"

namespace affineengine::autograd {

struct Tensor {
  affineengine::FlatStorage data;
  affineengine::FlatStorage grad;
  affineengine::Shape shape;
  bool requires_grad;
  bool apply_regularization;

  Tensor(const affineengine::Shape& init_shape, bool req_grad = true,
         bool apply_reg = false)
      : shape(init_shape),
        requires_grad(req_grad),
        apply_regularization(apply_reg) {
    Eigen::Index total_size = affineengine::compute_size(shape);
    data = affineengine::FlatStorage(total_size);
    data.setZero();

    if (requires_grad) {
      grad = affineengine::FlatStorage(total_size);
      grad.setZero();
    }
  }

  bool has_shape(const affineengine::Shape& target) const {
    return shape == target;
  }

  void resize(const affineengine::Shape& new_shape) {
    if (has_shape(new_shape)) return;
    shape = new_shape;
    Eigen::Index total_size = affineengine::compute_size(shape);
    data = affineengine::FlatStorage(total_size);
    if (requires_grad) {
      grad = affineengine::FlatStorage(total_size);
      grad.setZero();
    }
  }

  void zero_grad() {
    if (requires_grad) {
      grad.setZero();
    }
  }
};

}  // namespace affineengine::autograd