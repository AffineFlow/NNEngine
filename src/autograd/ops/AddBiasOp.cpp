#include "autograd/ops/AddBiasOp.hpp"

namespace affineengine::autograd::ops {
using MatrixMap = Eigen::Map<
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

void AddBiasOp::forward() {
  if (!out_->has_shape(a_->shape)) out_->resize(a_->shape);

  MatrixMap A_mat(a_->data.data(), a_->shape[0], a_->shape[1]);
  MatrixMap B_mat(b_->data.data(), 1, b_->shape[0]);
  MatrixMap Out_mat(out_->data.data(), out_->shape[0], out_->shape[1]);

  Out_mat.noalias() = A_mat.rowwise() + B_mat.row(0);
}

void AddBiasOp::backward() {
  MatrixMap dOut_mat(out_->grad.data(), out_->shape[0], out_->shape[1]);

  if (a_->requires_grad) {
    MatrixMap dA_mat(a_->grad.data(), a_->shape[0], a_->shape[1]);
    dA_mat.noalias() += dOut_mat;
  }
  if (b_->requires_grad) {
    MatrixMap dB_mat(b_->grad.data(), 1, b_->shape[0]);
    dB_mat.noalias() += dOut_mat.colwise().sum();
  }
}
}  // namespace affineengine::autograd::ops