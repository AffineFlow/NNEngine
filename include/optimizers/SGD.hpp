#pragma once

#include "core/Optimizer.hpp"

namespace affineengine::core {

/**
 * @brief Plain stochastic gradient descent optimizer.
 */
class SGD : public Optimizer {
 public:
  /**
   * @brief Construct SGD with the supplied learning rate.
   * @param learning_rate Step size used for parameter updates.
   */
  explicit SGD(float learning_rate = 0.01f);

  /**
   * @brief Apply the vanilla gradient descent update.
   */
  void step() override;
};

}  // namespace affineengine::core
