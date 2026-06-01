#pragma once

#include "autograd/Op.hpp"
#include "autograd/Tensor.hpp"

namespace mlengine::autograd::ops {

class DropoutOp : public Op {
  Tensor *a_, *out_;
  float p_;
  const bool* is_training_;
  mlengine::MatrixRM mask_;

 public:
  DropoutOp(Tensor* a, Tensor* out, float p, const bool* is_training)
      : a_(a), out_(out), p_(p), is_training_(is_training) {}

  void forward() override {
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
      float threshold = 2.0f * keep_prob - 1.0f;

      mask_.array() =
          (mlengine::MatrixRM::Random(a_->data.rows(), a_->data.cols())
               .array() < threshold)
              .select(mlengine::MatrixRM::Constant(a_->data.rows(),
                                                   a_->data.cols(), scale)
                          .array(),
                      mlengine::MatrixRM::Zero(a_->data.rows(), a_->data.cols())
                          .array());

      out_->data.noalias() = a_->data.cwiseProduct(mask_);
    }
  }

  void backward() override {
    if (!a_->requires_grad) return;

    if (*is_training_) {
      a_->grad.noalias() += out_->grad.cwiseProduct(mask_);
    } else {
      a_->grad.noalias() += out_->grad;
    }
  }
};

}  // namespace mlengine::autograd::ops