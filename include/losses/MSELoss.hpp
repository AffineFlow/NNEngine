#pragma once

#include "autograd/Tensor.hpp"
#include "core/Loss.hpp"

namespace affineengine::core {

/**
 * @brief Mean-squared-error objective for regression.
 */
class MSELoss : public Loss {
 public:
  float compute_loss() override;
  void backward() override;
};

}  // namespace affineengine::core