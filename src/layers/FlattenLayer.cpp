#include "layers/FlattenLayer.hpp"

#include "autograd/Tape.hpp"
#include "autograd/ops/FlattenOp.hpp"

namespace affineflow::layers {

autograd::Tensor* FlattenLayer::forward(autograd::Tensor* input) {
  auto* tape = autograd::Tape::get_global();
  bool req_grad = tape->record_ops_ && input->requires_grad;

  Eigen::Index batch = input->shape.empty() ? 1 : input->shape[0];
  Eigen::Index total = affineflow::compute_size(input->shape);
  Eigen::Index flat_dim = total / batch;

  auto* out = tape->alloc_tensor(affineflow::Shape{batch, flat_dim}, req_grad);
  auto* op = tape->allocate_op<autograd::ops::FlattenOp>(input, out);
  op->forward();

  return out;
}

}  // namespace affineflow::layers