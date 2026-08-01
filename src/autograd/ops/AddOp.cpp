#include "autograd/ops/AddOp.hpp"

#include <stdexcept>

namespace affineengine::autograd::ops {
void AddOp::forward() {
  if (a_->shape != b_->shape)
    throw std::invalid_argument("AddOp shape mismatch");
  if (!out_->has_shape(a_->shape)) out_->resize(a_->shape);
  out_->data = a_->data + b_->data;
}

void AddOp::backward() {
  if (a_->requires_grad) a_->grad += out_->grad;
  if (b_->requires_grad) b_->grad += out_->grad;
}
}  // namespace affineengine::autograd::ops