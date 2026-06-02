#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "autograd/ops/BatchNorm1dOp.hpp"
#include "core/Layer.hpp"

namespace mlengine::layers {

class BatchNorm1dLayer : public core::Layer {
  autograd::Tensor gamma_;
  autograd::Tensor beta_;
  // Wrap running stats in Tensors so they can be serialized via
  // named_parameters()
  autograd::Tensor running_mean_;
  autograd::Tensor running_var_;
  float momentum_, eps_;

 public:
  BatchNorm1dLayer(int num_features, float eps = 1e-5f, float momentum = 0.1f)
      : gamma_(core::MatrixRM::Ones(1, num_features), true),
        beta_(core::MatrixRM::Zero(1, num_features), true),
        running_mean_(core::MatrixRM::Zero(1, num_features),
                      false),  // requires_grad = false
        running_var_(core::MatrixRM::Ones(1, num_features),
                     false),  // requires_grad = false
        momentum_(momentum),
        eps_(eps) {}

  autograd::Tensor* forward(autograd::Tensor* input) override {
    auto* tape = autograd::Tape::get_global();
    bool req_grad =
        tape->record_ops_ &&
        (input->requires_grad || gamma_.requires_grad || beta_.requires_grad);
    auto* out =
        tape->alloc_tensor(input->data.rows(), input->data.cols(), req_grad);

    // Pass the underlying .data buffers to the Op so we don't have to change
    // the Op signature
    auto op = std::make_shared<autograd::ops::BatchNorm1dOp>(
        input, &gamma_, &beta_, out, &running_mean_.data, &running_var_.data,
        momentum_, eps_, &is_training_);
    op->forward();
    tape->record_op(op);
    return out;
  }

  std::vector<autograd::Tensor*> parameters() override {
    // Only return trainable parameters to the Optimizer
    return {&gamma_, &beta_};
  }

  std::map<std::string, autograd::Tensor*> named_parameters() override {
    // Return BOTH trainable weights and persistent state buffers for
    // Checkpointing
    return {{"weight", &gamma_},
            {"bias", &beta_},
            {"running_mean", &running_mean_},
            {"running_var", &running_var_}};
  }
};

}  // namespace mlengine::layers