#include "losses/SoftmaxCrossEntropyLoss.hpp"

namespace affineflow::nn::core {
using ArrayMap = Eigen::Map<
    Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

float SoftmaxCrossEntropyLoss::compute_loss() {
  int batch_size = predictions_->shape[0];
  int features = predictions_->shape[1];

  ArrayMap pred_map(predictions_->data.data(), batch_size, features);
  ArrayMap targ_map(targets_->data.data(), batch_size, features);

  Eigen::ArrayXf max_vals = pred_map.rowwise().maxCoeff();
  Eigen::ArrayXXf shifted = pred_map.colwise() - max_vals;
  Eigen::ArrayXf sums = shifted.exp().rowwise().sum();
  Eigen::ArrayXf log_sums = sums.log();

  return -(targ_map * (shifted.colwise() - log_sums)).sum() /
         static_cast<float>(batch_size);
}

void SoftmaxCrossEntropyLoss::backward() {
  if (predictions_->requires_grad) {
    int batch_size = predictions_->shape[0];
    int features = predictions_->shape[1];

    ArrayMap pred_map(predictions_->data.data(), batch_size, features);
    ArrayMap targ_map(targets_->data.data(), batch_size, features);
    ArrayMap grad_map(predictions_->grad.data(), batch_size, features);

    Eigen::ArrayXf max_vals = pred_map.rowwise().maxCoeff();
    Eigen::ArrayXXf shifted = pred_map.colwise() - max_vals;
    Eigen::ArrayXf sums = shifted.exp().rowwise().sum();

    grad_map += (shifted.exp().colwise() / sums - targ_map) /
                static_cast<float>(batch_size);
  }
}
}  // namespace affineflow::nn::core