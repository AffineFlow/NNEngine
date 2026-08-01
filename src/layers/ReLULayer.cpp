#include "layers/ReLULayer.hpp"

#include "autograd/Tape.hpp"
#include "autograd/ops/ReLUOp.hpp"

namespace affineflow::layers {

autograd::Tensor* ReLULayer::forward(autograd::Tensor* input) {
  auto* tape = autograd::Tape::get_global();
  bool req_grad = tape->record_ops_ && input->requires_grad;
  auto* out = tape->alloc_tensor(input->shape, req_grad);
  auto* op = tape->allocate_op<affineflow::autograd::ops::ReLUOp>(input, out);
  op->forward();
  return out;
}

}  // namespace affineflow::layers