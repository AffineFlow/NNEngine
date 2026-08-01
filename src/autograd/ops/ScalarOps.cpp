#include "autograd/ops/ScalarOps.hpp"

namespace affineengine::autograd::ops {

void AddScalarOp::forward() {
  if (!out_->has_shape(a_->shape)) out_->resize(a_->shape);
  out_->data = a_->data + val_;
}
void AddScalarOp::backward() {
  if (a_->requires_grad) a_->grad += out_->grad;
}

void SubScalarOp::forward() {
  if (!out_->has_shape(a_->shape)) out_->resize(a_->shape);
  out_->data = a_->data - val_;
}
void SubScalarOp::backward() {
  if (a_->requires_grad) a_->grad += out_->grad;
}

void RSubScalarOp::forward() {
  if (!out_->has_shape(a_->shape)) out_->resize(a_->shape);
  out_->data = val_ - a_->data;
}
void RSubScalarOp::backward() {
  if (a_->requires_grad) a_->grad -= out_->grad;
}

void MulScalarOp::forward() {
  if (!out_->has_shape(a_->shape)) out_->resize(a_->shape);
  out_->data = a_->data * val_;
}
void MulScalarOp::backward() {
  if (a_->requires_grad) a_->grad += out_->grad * val_;
}

void DivScalarOp::forward() {
  if (!out_->has_shape(a_->shape)) out_->resize(a_->shape);
  out_->data = a_->data / val_;
}
void DivScalarOp::backward() {
  if (a_->requires_grad) a_->grad += out_->grad / val_;
}

void RDivScalarOp::forward() {
  if (!out_->has_shape(a_->shape)) out_->resize(a_->shape);
  out_->data = val_ / a_->data;
}
void RDivScalarOp::backward() {
  if (a_->requires_grad) {
    a_->grad -= out_->grad * val_ / (a_->data * a_->data);
  }
}

}  // namespace affineengine::autograd::ops