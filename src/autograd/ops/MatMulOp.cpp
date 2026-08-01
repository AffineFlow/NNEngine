#include "autograd/ops/MatMulOp.hpp"

namespace affineengine::autograd::ops {
using MatrixMap = Eigen::Map<
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

void MatMulOp::forward() {
  affineengine::Shape out_shape = {a_->shape[0], b_->shape[1]};
  if (!out_->has_shape(out_shape)) out_->resize(out_shape);

  MatrixMap A_mat(a_->data.data(), a_->shape[0], a_->shape[1]);
  MatrixMap B_mat(b_->data.data(), b_->shape[0], b_->shape[1]);
  MatrixMap Out_mat(out_->data.data(), out_shape[0], out_shape[1]);

  Out_mat.noalias() = A_mat * B_mat;
}

void MatMulOp::backward() {
  MatrixMap A_mat(a_->data.data(), a_->shape[0], a_->shape[1]);
  MatrixMap B_mat(b_->data.data(), b_->shape[0], b_->shape[1]);
  MatrixMap dOut_mat(out_->grad.data(), out_->shape[0], out_->shape[1]);

  if (a_->requires_grad) {
    MatrixMap dA_mat(a_->grad.data(), a_->shape[0], a_->shape[1]);
    dA_mat.noalias() += dOut_mat * B_mat.transpose();
  }
  if (b_->requires_grad) {
    MatrixMap dB_mat(b_->grad.data(), b_->shape[0], b_->shape[1]);
    dB_mat.noalias() += A_mat.transpose() * dOut_mat;
  }
}
}  // namespace affineengine::autograd::ops