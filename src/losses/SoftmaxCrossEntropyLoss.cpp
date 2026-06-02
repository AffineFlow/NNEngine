#include "losses/SoftmaxCrossEntropyLoss.hpp"

namespace mlengine::core {

float SoftmaxCrossEntropyLoss::compute_loss() {
  int batch_size = predictions_->data.rows();

  Eigen::VectorXf max_vals = predictions_->data.rowwise().maxCoeff();
  mlengine::MatrixRM shifted = predictions_->data.colwise() - max_vals;
  Eigen::VectorXf sums = shifted.array().exp().rowwise().sum();
  Eigen::VectorXf log_sums = sums.array().log();

  return -(targets_->data.array() * (shifted.colwise() - log_sums).array())
              .sum() /
         static_cast<float>(batch_size);
}

void SoftmaxCrossEntropyLoss::backward() {
  if (predictions_->requires_grad) {
    int batch_size = predictions_->data.rows();
    Eigen::VectorXf max_vals = predictions_->data.rowwise().maxCoeff();
    mlengine::MatrixRM shifted = predictions_->data.colwise() - max_vals;
    Eigen::VectorXf sums = shifted.array().exp().rowwise().sum();

    predictions_->grad.array() +=
        (shifted.array().exp().colwise() / sums.array() -
         targets_->data.array()) /
        static_cast<float>(batch_size);
  }
}

}  // namespace mlengine::core