#pragma once
#include <utility>

#include "core/Types.hpp"

namespace affineflow::nn::autograd {

struct Tensor {
  affineflow::nn::FlatStorage data;
  affineflow::nn::FlatStorage grad;
  affineflow::nn::Shape shape;
  bool requires_grad;
  bool apply_regularization;

  Tensor(const affineflow::nn::Shape& init_shape, bool req_grad = true,
         bool apply_reg = false)
      : shape(init_shape),
        requires_grad(req_grad),
        apply_regularization(apply_reg) {
    Eigen::Index total_size = affineflow::nn::compute_size(shape);
    data = affineflow::nn::FlatStorage(total_size);
    data.setZero();

    if (requires_grad) {
      grad = affineflow::nn::FlatStorage(total_size);
      grad.setZero();
    }
  }

  bool has_shape(const affineflow::nn::Shape& target) const {
    return shape == target;
  }

  // Safely provisions the gradient buffer if it is missing
  void allocate_grad() {
    if (grad.size() != data.size()) {
      grad = affineflow::nn::FlatStorage(data.dimensions());
      grad.setZero();
    }
  }

  void resize(const affineflow::nn::Shape& new_shape) {
    if (!has_shape(new_shape)) {
      shape = new_shape;
      Eigen::Index total_size = affineflow::nn::compute_size(shape);
      data = affineflow::nn::FlatStorage(total_size);
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

}  // namespace affineflow::nn::autograd