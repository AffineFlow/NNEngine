#include "autograd/ops/BatchNorm1dOp.hpp"

namespace mlengine::autograd::ops {

BatchNorm1dOp::BatchNorm1dOp(Tensor* x, Tensor* gamma, Tensor* beta,
                             Tensor* out, mlengine::MatrixRM* running_mean,
                             mlengine::MatrixRM* running_var, float momentum,
                             float eps, const bool* is_training)
    : x_(x),
      gamma_(gamma),
      beta_(beta),
      out_(out),
      running_mean_(running_mean),
      running_var_(running_var),
      momentum_(momentum),
      eps_(eps),
      is_training_(is_training) {}

void BatchNorm1dOp::forward() {
  int batch = x_->data.rows();
  int feat = x_->data.cols();

  if (out_->data.rows() != batch || out_->data.cols() != feat) {
    out_->data.resize(batch, feat);
    if (out_->requires_grad) {
      out_->grad.resize(batch, feat);
      out_->grad.setZero();
    }
  }

  if (*is_training_) {
    mlengine::MatrixRM batch_mean = x_->data.colwise().mean();
    x_centered_ = x_->data.rowwise() - batch_mean.row(0);
    mlengine::MatrixRM batch_var =
        x_centered_.array().square().colwise().mean();
    stddev_inv_ = (batch_var.array() + eps_).inverse().sqrt();

    x_hat_ = x_centered_.array().rowwise() * stddev_inv_.row(0).array();

    *running_mean_ = (1.0f - momentum_) * running_mean_->array() +
                     momentum_ * batch_mean.array();
    float unbias = batch > 1 ? static_cast<float>(batch) / (batch - 1) : 1.0f;
    *running_var_ = (1.0f - momentum_) * running_var_->array() +
                    momentum_ * batch_var.array() * unbias;

    out_->data =
        (x_hat_.array().rowwise() * gamma_->data.row(0).array()).rowwise() +
        beta_->data.row(0).array();
  } else {
    mlengine::MatrixRM std_inv =
        (running_var_->array() + eps_).inverse().sqrt();
    mlengine::MatrixRM x_hat_eval =
        (x_->data.rowwise() - running_mean_->row(0)).array().rowwise() *
        std_inv.row(0).array();
    out_->data =
        (x_hat_eval.array().rowwise() * gamma_->data.row(0).array()).rowwise() +
        beta_->data.row(0).array();
  }
}

void BatchNorm1dOp::backward() {
  if (!*is_training_) return;
  int batch = x_->data.rows();

  if (gamma_->requires_grad) {
    gamma_->grad.row(0) += (out_->grad.cwiseProduct(x_hat_)).colwise().sum();
  }
  if (beta_->requires_grad) {
    beta_->grad.row(0) += out_->grad.colwise().sum();
  }

  if (x_->requires_grad) {
    mlengine::MatrixRM dx_hat =
        out_->grad.array().rowwise() * gamma_->data.row(0).array();
    mlengine::MatrixRM dx_hat_sum = dx_hat.colwise().sum();
    mlengine::MatrixRM dx_hat_x_hat_sum =
        (dx_hat.cwiseProduct(x_hat_)).colwise().sum();

    mlengine::MatrixRM term =
        (dx_hat * static_cast<float>(batch)).rowwise() - dx_hat_sum.row(0);
    term.array() -= x_hat_.array().rowwise() * dx_hat_x_hat_sum.row(0).array();

    x_->grad.array() +=
        term.array().rowwise() * (stddev_inv_.row(0).array() * (1.0f / batch));
  }
}

}  // namespace mlengine::autograd::ops