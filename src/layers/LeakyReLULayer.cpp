#include "layers/LeakyReLULayer.hpp"

#include "autograd/Tape.hpp"
#include "autograd/ops/LeakyReLUOp.hpp"

namespace affineengine::layers {

LeakyReLULayer::LeakyReLULayer(float alpha) : alpha_(alpha) {}

autograd::Tensor* LeakyReLULayer::forward(autograd::Tensor* input) {
  auto* tape = autograd::Tape::get_global();
  bool req_grad = tape->record_ops_ && input->requires_grad;
  auto* out = tape->alloc_tensor(input->shape, req_grad);
  auto* op = tape->allocate_op<affineengine::autograd::ops::LeakyReLUOp>(input, out,
                                                                     alpha_);
  op->forward();
  return out;
}

}  // namespace affineengine::layers