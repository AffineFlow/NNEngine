#pragma once
#include <utility>

#include "core/Types.hpp"

namespace affineflow::autograd {

struct Tensor {
  affineflow::FlatStorage data;
  affineflow::FlatStorage grad;
  affineflow::Shape shape;
  bool requires_grad;
  bool apply_regularization;

  Tensor(const affineflow::Shape& init_shape, bool req_grad = true,
         bool apply_reg = false)
      : shape(init_shape),
        requires_grad(req_grad),
        apply_regularization(apply_reg) {
    Eigen::Index total_size = affineflow::compute_size(shape);
    data = affineflow::FlatStorage(total_size);
    data.setZero();

    if (requires_grad) {
      grad = affineflow::FlatStorage(total_size);
      grad.setZero();
    }
  }

  bool has_shape(const affineflow::Shape& target) const {
    return shape == target;
  }

  // Safely provisions the gradient buffer if it is missing
  void allocate_grad() {
    if (grad.size() != data.size()) {
      grad = affineflow::FlatStorage(data.dimensions());
      grad.setZero();
    }
  }

  void resize(const affineflow::Shape& new_shape) {
    if (!has_shape(new_shape)) {
      shape = new_shape;
      Eigen::Index total_size = affineflow::compute_size(shape);
      data = affineflow::FlatStorage(total_size);
      if (requires_grad) {
        allocate_grad();
      }
    } else if (requires_grad) {
      allocate_grad();
    }
  }

  void zero_grad() {
    if (requires_grad) {
      allocate_grad();
      grad.setZero();
    }
  }
};

}  // namespace affineflow::autograd