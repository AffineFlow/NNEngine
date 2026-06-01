#pragma once
#include <memory>

#include "autograd/ops/LeakyReLUOp.hpp"
#include "core/Layer.hpp"

namespace mlengine::layers {

/**
 * @brief Elementwise leaky rectified linear activation.
 */
class LeakyReLULayer : public core::Layer {
 public:
  /**
   * @brief Construct a leaky ReLU layer with the given negative slope.
   * @param alpha Slope applied to negative activations.
   */
  LeakyReLULayer(float alpha = 0.01f) : alpha_(alpha) {}

  /**
   * @brief Apply leaky ReLU to the incoming activation tensor.
   * @param input Input activation tensor.
   * @return Pointer to the tape-owned output tensor.
   * @note The returned tensor is arena-allocated and owned by the tape, not
   *     the caller.
   */
  autograd::Tensor* forward(autograd::Tensor* input) override {
    auto* tape = autograd::Tape::get_global();
    bool req_grad = tape->record_ops_ && input->requires_grad;
    auto* out =
        tape->alloc_tensor(input->data.rows(), input->data.cols(), req_grad);
    auto op = std::make_shared<autograd::ops::LeakyReLUOp>(input, out, alpha_);
    op->forward();
    tape->record_op(op);
    return out;
  }

 private:
  float alpha_;
};

}  // namespace mlengine::layers