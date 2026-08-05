#include "layers/DropoutLayer.hpp"

#include "autograd/Tape.hpp"
#include "autograd/ops/DropoutOp.hpp"

namespace affineflow::nn::layers {

DropoutLayer::DropoutLayer(float p) : p_(p) {}

autograd::Tensor* DropoutLayer::forward(autograd::Tensor* input) {
  auto* tape = autograd::Tape::get_global();
  bool req_grad = tape->record_ops_ && input->requires_grad;
  auto* out = tape->alloc_tensor(input->shape, req_grad);
  auto* op = tape->allocate_op<affineflow::nn::autograd::ops::DropoutOp>(
      input, out, p_, &is_training_);
  op->forward();
  return out;
}

}  // namespace affineflow::nn::layers