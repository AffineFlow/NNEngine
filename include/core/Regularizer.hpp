#pragma once

#include <vector>

#include "autograd/Tensor.hpp"

namespace affineflow::nn::core {

/**
 * @brief Penalty term applied to trainable parameters during optimization.
 */
class Regularizer {
 public:
  virtual ~Regularizer() = default;

  /**
   * @brief Accumulate a regularization penalty and any gradient adjustment.
   * @param parameters Trainable tensors owned by the model.
   * @return Scalar regularization penalty to add to the loss.
   */
  virtual float apply(const std::vector<autograd::Tensor*>& parameters) = 0;
};

}  // namespace affineflow::nn::core
