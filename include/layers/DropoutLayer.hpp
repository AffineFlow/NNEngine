#pragma once
#include <memory>

#include "autograd/ops/DropoutOp.hpp"
#include "core/Layer.hpp"

namespace mlengine::layers {

class DropoutLayer : public core::Layer {
 public:
  float p_;

  /**
   * @brief Create a Dropout layer.
   * @param p Probability of an element to be zeroed. Default: 0.5
   */
  explicit DropoutLayer(float p = 0.5f) : p_(p) {}

  autograd::Tensor* forward(autograd::Tensor* input) override {
    auto* tape = autograd::Tape::get_global();
    bool req_grad = tape->record_ops_ && input->requires_grad;
    auto* out =
        tape->alloc_tensor(input->data.rows(), input->data.cols(), req_grad);

    auto op = std::make_shared<autograd::ops::DropoutOp>(input, out, p_,
                                                         &is_training_);
    op->forward();
    tape->record_op(op);

    return out;
  }
};

}  // namespace mlengine::layers