#include "autograd/ops/DropoutOp.hpp"

#include <random>

#include "core/Random.hpp"

namespace mlengine::autograd::ops {
DropoutOp::DropoutOp(Tensor* a, Tensor* out, float p, const bool* is_training)
    : a_(a), out_(out), p_(p), is_training_(is_training) {}

void DropoutOp::forward() {
  if (!out_->has_shape(a_->shape)) out_->resize(a_->shape);

  if (!(*is_training_)) {
    out_->data = a_->data;
  } else {
    if (mask_.rows() != 1 || mask_.cols() != a_->data.size()) {
      mask_.resize(1, a_->data.size());
    }

    float keep_prob = 1.0f - p_;
    float scale = 1.0f / keep_prob;

    // Vectorized mask generation using Eigen's RNG
    float threshold = -1.0f + 2.0f * keep_prob;
    Eigen::Map<Eigen::ArrayXf> mask_arr(mask_.data(), a_->data.size());
    mask_arr =
        (Eigen::ArrayXf::Random(a_->data.size()) < threshold).cast<float>() *
        scale;

    Eigen::TensorMap<mlengine::FlatStorage> mask_map(mask_.data(),
                                                     a_->data.dimensions());
    out_->data = a_->data * mask_map;
  }
}

void DropoutOp::backward() {
  if (!a_->requires_grad) return;
  if (*is_training_) {
    Eigen::TensorMap<mlengine::FlatStorage> mask_map(mask_.data(),
                                                     a_->data.dimensions());
    a_->grad += out_->grad * mask_map;
  } else {
    a_->grad += out_->grad;
  }
}
}  // namespace mlengine::autograd::ops