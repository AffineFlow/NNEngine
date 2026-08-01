#include "autograd/ops/DivOp.hpp"

#include <stdexcept>

namespace affineengine::autograd::ops {
void DivOp::forward() {
  if (a_->shape != b_->shape)
    throw std::invalid_argument("DivOp shape mismatch");
  if (!out_->has_shape(a_->shape)) out_->resize(a_->shape);
  out_->data = a_->data / b_->data;
}

void DivOp::backward() {
  if (a_->requires_grad) a_->grad += out_->grad / b_->data;
  if (b_->requires_grad)
    b_->grad -= out_->grad * a_->data / (b_->data * b_->data);
}
}  // namespace affineengine::autograd::ops