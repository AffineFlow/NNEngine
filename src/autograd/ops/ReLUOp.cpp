#include "autograd/ops/ReLUOp.hpp"

namespace affineflow::autograd::ops {
void ReLUOp::forward() {
  if (!out_->has_shape(a_->shape)) out_->resize(a_->shape);
  out_->data = a_->data.cwiseMax(0.0f);
}

void ReLUOp::backward() {
  if (a_->requires_grad) {
    a_->grad +=
        (a_->data > 0.0f)
            .select(out_->grad,
                    affineflow::FlatStorage(a_->data.dimensions()).setZero());
  }
}
}  // namespace affineflow::autograd::ops