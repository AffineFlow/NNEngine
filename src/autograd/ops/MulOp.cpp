#include "autograd/ops/MulOp.hpp"

#include <stdexcept>

namespace affineflow::nn::autograd::ops {
void MulOp::forward() {
  if (a_->shape != b_->shape)
    throw std::invalid_argument("MulOp shape mismatch");
  if (!out_->has_shape(a_->shape)) out_->resize(a_->shape);
  out_->data = a_->data * b_->data;
}

void MulOp::backward() {
  if (a_->requires_grad) a_->grad += out_->grad * b_->data;
  if (b_->requires_grad) b_->grad += out_->grad * a_->data;
}
}  // namespace affineflow::nn::autograd::ops