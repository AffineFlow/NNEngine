#include "layers/DropoutLayer.hpp"

#include <memory>

#include "autograd/Tape.hpp"
#include "autograd/ops/DropoutOp.hpp"

namespace mlengine::layers {

DropoutLayer::DropoutLayer(float p) : p_(p) {}

autograd::Tensor* DropoutLayer::forward(autograd::Tensor* input) {
  auto* tape = autograd::Tape::get_global();
  bool req_grad = tape->record_ops_ && input->requires_grad;
  auto* out =
      tape->alloc_tensor(input->data.rows(), input->data.cols(), req_grad);

  auto op =
      std::make_shared<autograd::ops::DropoutOp>(input, out, p_, &is_training_);
  op->forward();
  tape->record_op(op);

  return out;
}

}  // namespace mlengine::layers