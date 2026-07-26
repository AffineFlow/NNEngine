#include "autograd/ops/ReductionOps.hpp"

#include <Eigen/Core>

namespace mlengine::autograd::ops {

void SumOp::forward() {
  if (!out_->has_shape({1})) out_->resize({1});
  Eigen::Map<Eigen::ArrayXf> a_arr(a_->data.data(), a_->data.size());
  out_->data.data()[0] = a_arr.sum();
}

void SumOp::backward() {
  if (a_->requires_grad) {
    float g = out_->grad.data()[0];
    a_->grad = a_->grad + g;
  }
}

void MeanOp::forward() {
  if (!out_->has_shape({1})) out_->resize({1});
  Eigen::Map<Eigen::ArrayXf> a_arr(a_->data.data(), a_->data.size());
  out_->data.data()[0] = a_arr.mean();
}

void MeanOp::backward() {
  if (a_->requires_grad) {
    float g = out_->grad.data()[0] / static_cast<float>(a_->data.size());
    a_->grad = a_->grad + g;
  }
}

}  // namespace mlengine::autograd::ops