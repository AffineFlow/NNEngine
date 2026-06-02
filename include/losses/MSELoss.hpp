#pragma once

#include "autograd/Tensor.hpp"
#include "core/Loss.hpp"

namespace mlengine::core {

/**
 * @brief Mean-squared-error objective for regression.
 */
class MSELoss : public Loss {
 public:
  float compute_loss() override;
  void backward() override;
};

}  // namespace mlengine::core