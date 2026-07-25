#include "losses/MSELoss.hpp"

namespace mlengine::core {
float MSELoss::compute_loss() {
  Eigen::Map<Eigen::ArrayXf> pred_map(predictions_->data.data(),
                                      predictions_->data.size());
  Eigen::Map<Eigen::ArrayXf> targ_map(targets_->data.data(),
                                      targets_->data.size());
  return (pred_map - targ_map).square().sum() /
         static_cast<float>(predictions_->shape[0]);
}

void MSELoss::backward() {
  if (predictions_->requires_grad) {
    predictions_->grad += 2.0f * (predictions_->data - targets_->data) /
                          static_cast<float>(predictions_->shape[0]);
  }
}
}  // namespace mlengine::core