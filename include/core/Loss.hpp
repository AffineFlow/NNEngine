#pragma once

#include "autograd/Tensor.hpp"
#include "core/Types.hpp"

namespace mlengine::core {

/**
 * @brief Objective function that produces a scalar training signal.
 */
class Loss {
 protected:
  autograd::Tensor* predictions_ = nullptr;
  autograd::Tensor* targets_ = nullptr;

 public:
  virtual ~Loss() = default;

  /**
   * @brief Bind tensors and compute the forward loss.
   */
  virtual float forward(autograd::Tensor* predictions,
                        autograd::Tensor* targets) {
    predictions_ = predictions;
    targets_ = targets;
    return compute_loss();
  }

  /**
   * @brief Pure virtual function for mathematical loss calculation.
   */
  virtual float compute_loss() = 0;

  /**
   * @brief Explicitly seed the gradient into predictions_->grad.
   */
  virtual void backward() = 0;
};

}  // namespace mlengine::core