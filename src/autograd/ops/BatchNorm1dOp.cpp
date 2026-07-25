#include "autograd/ops/BatchNorm1dOp.hpp"

namespace mlengine::autograd::ops {
using ArrayMap = Eigen::Map<
    Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

BatchNorm1dOp::BatchNorm1dOp(Tensor* x, Tensor* gamma, Tensor* beta,
                             Tensor* out, Tensor* running_mean,
                             Tensor* running_var, float momentum, float eps,
                             const bool* is_training)
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
  if (!out_->has_shape(x_->shape)) out_->resize(x_->shape);

  int batch = x_->shape[0];
  int feat = x_->shape[1];

  if (x_centered_.size() != batch * feat) x_centered_.resize(batch * feat);
  if (stddev_inv_.size() != feat) stddev_inv_.resize(feat);
  if (x_hat_.size() != batch * feat) x_hat_.resize(batch * feat);

  ArrayMap x_arr(x_->data.data(), batch, feat);
  ArrayMap gamma_arr(gamma_->data.data(), 1, feat);
  ArrayMap beta_arr(beta_->data.data(), 1, feat);
  ArrayMap out_arr(out_->data.data(), batch, feat);
  Eigen::Map<Eigen::ArrayXf> r_mean(running_mean_->data.data(), feat);
  Eigen::Map<Eigen::ArrayXf> r_var(running_var_->data.data(), feat);

  if (*is_training_) {
    Eigen::ArrayXf batch_mean = x_arr.colwise().mean();

    ArrayMap xc_arr(x_centered_.data(), batch, feat);
    xc_arr = x_arr.rowwise() - batch_mean.transpose();

    Eigen::ArrayXf batch_var = xc_arr.square().colwise().mean();
    Eigen::Map<Eigen::ArrayXf> std_inv(stddev_inv_.data(), feat);
    std_inv = (batch_var + eps_).inverse().sqrt();

    ArrayMap xh_arr(x_hat_.data(), batch, feat);
    xh_arr = xc_arr.rowwise() * std_inv.transpose();

    r_mean = (1.0f - momentum_) * r_mean + momentum_ * batch_mean;
    float unbias = batch > 1 ? static_cast<float>(batch) / (batch - 1) : 1.0f;
    r_var = (1.0f - momentum_) * r_var + momentum_ * batch_var * unbias;

    out_arr = (xh_arr.rowwise() * gamma_arr.row(0)).rowwise() + beta_arr.row(0);
  } else {
    Eigen::ArrayXf std_inv = (r_var + eps_).inverse().sqrt();
    Eigen::ArrayXXf x_hat_eval =
        (x_arr.rowwise() - r_mean.transpose()).rowwise() * std_inv.transpose();
    out_arr =
        (x_hat_eval.rowwise() * gamma_arr.row(0)).rowwise() + beta_arr.row(0);
  }
}

void BatchNorm1dOp::backward() {
  if (!*is_training_) return;
  int batch = x_->shape[0];
  int feat = x_->shape[1];

  ArrayMap dx_arr(x_->grad.data(), batch, feat);
  ArrayMap dgamma_arr(gamma_->grad.data(), 1, feat);
  ArrayMap dbeta_arr(beta_->grad.data(), 1, feat);
  ArrayMap dout_arr(out_->grad.data(), batch, feat);
  ArrayMap gamma_arr(gamma_->data.data(), 1, feat);

  ArrayMap xh_arr(x_hat_.data(), batch, feat);

  if (gamma_->requires_grad) {
    dgamma_arr.row(0) += (dout_arr * xh_arr).colwise().sum();
  }
  if (beta_->requires_grad) {
    dbeta_arr.row(0) += dout_arr.colwise().sum();
  }
  if (x_->requires_grad) {
    Eigen::ArrayXXf dx_hat = dout_arr.rowwise() * gamma_arr.row(0);
    Eigen::ArrayXf dx_hat_sum = dx_hat.colwise().sum();
    Eigen::ArrayXf dx_hat_x_hat_sum = (dx_hat * xh_arr).colwise().sum();

    Eigen::ArrayXXf term =
        (dx_hat * static_cast<float>(batch)).rowwise() - dx_hat_sum.transpose();
    term -= xh_arr.rowwise() * dx_hat_x_hat_sum.transpose();

    Eigen::Map<Eigen::ArrayXf> std_inv(stddev_inv_.data(), feat);
    dx_arr += term.rowwise() * (std_inv.transpose() * (1.0f / batch));
  }
}
}  // namespace mlengine::autograd::ops