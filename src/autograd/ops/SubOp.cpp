#include "autograd/ops/SubOp.hpp"

#include <stdexcept>

namespace affineflow::autograd::ops {
void SubOp::forward() {
  if (a_->shape != b_->shape)
    throw std::invalid_argument("SubOp shape mismatch");
  if (!out_->has_shape(a_->shape)) out_->resize(a_->shape);
  out_->data = a_->data - b_->data;
}

void SubOp::backward() {
  if (a_->requires_grad) a_->grad += out_->grad;
  if (b_->requires_grad) b_->grad -= out_->grad;
}
}  // namespace affineflow::autograd::ops