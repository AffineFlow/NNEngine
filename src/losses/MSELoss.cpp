#include "losses/MSELoss.hpp"

namespace mlengine::core {

float MSELoss::compute_loss() {
  return (predictions_->data - targets_->data).squaredNorm() /
         static_cast<float>(predictions_->data.rows());
}

void MSELoss::backward() {
  if (predictions_->requires_grad) {
    predictions_->grad.noalias() +=
        2.0f * (predictions_->data - targets_->data) /
        static_cast<float>(predictions_->data.rows());
  }
}

}  // namespace mlengine::core