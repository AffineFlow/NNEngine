#pragma once
#include <memory>

#include "autograd/ops/ReLUOp.hpp"
#include "core/Layer.hpp"

namespace mlengine::layers {

/**
 * @brief Elementwise rectified linear activation.
 */
class ReLULayer : public core::Layer {
 public:
  /**
   * @brief Apply ReLU to the incoming activation tensor.
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
    auto op = std::make_shared<autograd::ops::ReLUOp>(input, out);
    op->forward();
    tape->record_op(op);
    return out;
  }
};

}  // namespace mlengine::layers