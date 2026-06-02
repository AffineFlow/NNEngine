#include "autograd/ops/DropoutOp.hpp"

#include <random>

#include "core/Random.hpp"

namespace mlengine::autograd::ops {

DropoutOp::DropoutOp(Tensor* a, Tensor* out, float p, const bool* is_training)
    : a_(a), out_(out), p_(p), is_training_(is_training) {}

void DropoutOp::forward() {
  if (out_->data.rows() != a_->data.rows() ||
      out_->data.cols() != a_->data.cols()) {
    out_->data.resize(a_->data.rows(), a_->data.cols());
    if (out_->requires_grad) {
      out_->grad.resize(a_->data.rows(), a_->data.cols());
      out_->grad.setZero();
    }
  }

  if (!(*is_training_)) {
    out_->data.noalias() = a_->data;
  } else {
    if (mask_.rows() != a_->data.rows() || mask_.cols() != a_->data.cols()) {
      mask_.resize(a_->data.rows(), a_->data.cols());
    }

    float keep_prob = 1.0f - p_;
    float scale = 1.0f / keep_prob;

    std::bernoulli_distribution d(keep_prob);
    auto& gen = core::rng();

    float* mask_ptr = mask_.data();
    size_t size = mask_.size();
    for (size_t i = 0; i < size; ++i) {
      mask_ptr[i] = d(gen) ? scale : 0.0f;
    }
    out_->data.noalias() = a_->data.cwiseProduct(mask_);
  }
}

void DropoutOp::backward() {
  if (!a_->requires_grad) return;

  if (*is_training_) {
    a_->grad.noalias() += out_->grad.cwiseProduct(mask_);
  } else {
    a_->grad.noalias() += out_->grad;
  }
}

}  // namespace mlengine::autograd::ops