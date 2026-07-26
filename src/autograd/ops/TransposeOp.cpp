#include "autograd/ops/TransposeOp.hpp"

#include <Eigen/Core>
#include <stdexcept>

namespace mlengine::autograd::ops {

using MatrixMap = Eigen::Map<
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

void TransposeOp::forward() {
  if (a_->shape.size() != 2) {
    throw std::runtime_error("TransposeOp currently only supports 2D tensors.");
  }

  if (!out_->has_shape({a_->shape[1], a_->shape[0]})) {
    out_->resize({a_->shape[1], a_->shape[0]});
  }

  MatrixMap a_mat(a_->data.data(), a_->shape[0], a_->shape[1]);
  MatrixMap out_mat(out_->data.data(), out_->shape[0], out_->shape[1]);

  out_mat = a_mat.transpose();
}

void TransposeOp::backward() {
  if (a_->requires_grad) {
    MatrixMap grad_out_mat(out_->grad.data(), out_->shape[0], out_->shape[1]);
    MatrixMap grad_a_mat(a_->grad.data(), a_->shape[0], a_->shape[1]);

    grad_a_mat += grad_out_mat.transpose();
  }
}

}  // namespace mlengine::autograd::ops