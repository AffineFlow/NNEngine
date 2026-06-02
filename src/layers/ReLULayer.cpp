#include "layers/ReLULayer.hpp"

#include <memory>

#include "autograd/Tape.hpp"
#include "autograd/ops/ReLUOp.hpp"

namespace mlengine::layers {

autograd::Tensor* ReLULayer::forward(autograd::Tensor* input) {
  auto* tape = autograd::Tape::get_global();
  bool req_grad = tape->record_ops_ && input->requires_grad;
  auto* out =
      tape->alloc_tensor(input->data.rows(), input->data.cols(), req_grad);
  auto op = std::make_shared<autograd::ops::ReLUOp>(input, out);
  op->forward();
  tape->record_op(op);
  return out;
}

}  // namespace mlengine::layers