#include "autograd/ops/LeakyReLUOp.hpp"

namespace affineflow::autograd::ops {
void LeakyReLUOp::forward() {
  if (!out_->has_shape(a_->shape)) out_->resize(a_->shape);
  out_->data = (a_->data > 0.0f).select(a_->data, a_->data * alpha_);
}

void LeakyReLUOp::backward() {
  if (a_->requires_grad) {
    a_->grad += (a_->data > 0.0f).select(out_->grad, out_->grad * alpha_);
  }
}
}  // namespace affineflow::autograd::ops