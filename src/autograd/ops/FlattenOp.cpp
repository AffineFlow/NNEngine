#include "autograd/ops/FlattenOp.hpp"

#include "core/Types.hpp"

namespace affineflow::nn::autograd::ops {

void FlattenOp::forward() {
  if (x_->shape.empty()) return;

  Eigen::Index batch = x_->shape[0];
  Eigen::Index total = affineflow::nn::compute_size(x_->shape);
  Eigen::Index flat_dim = total / batch;

  if (!out_->has_shape({batch, flat_dim})) {
    out_->resize({batch, flat_dim});
  }

  // Flat memory makes this a fully vectorized 1:1 copy
  out_->data = x_->data;
}

void FlattenOp::backward() {
  if (x_->requires_grad) {
    // 1:1 gradient routing
    x_->grad += out_->grad;
  }
}

}  // namespace affineflow::nn::autograd::ops