#include "autograd/ops/UnaryOps.hpp"

#include <Eigen/Core>

namespace affineengine::autograd::ops {

void ExpOp::forward() {
  if (!out_->has_shape(a_->shape)) out_->resize(a_->shape);
  out_->data = a_->data.exp();
}
void ExpOp::backward() {
  if (a_->requires_grad) {
    a_->grad += out_->grad * out_->data;  // d/dx e^x = e^x
  }
}

void LogOp::forward() {
  if (!out_->has_shape(a_->shape)) out_->resize(a_->shape);
  out_->data = a_->data.log();
}
void LogOp::backward() {
  if (a_->requires_grad) {
    a_->grad += out_->grad / a_->data;  // d/dx ln(x) = 1/x
  }
}

}  // namespace affineengine::autograd::ops