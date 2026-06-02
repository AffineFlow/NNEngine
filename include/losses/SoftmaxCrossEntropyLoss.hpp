#pragma once

#include "autograd/Tensor.hpp"
#include "core/Loss.hpp"

namespace mlengine::core {

/**
 * @brief Numerically stable softmax cross-entropy for classification.
 */
class SoftmaxCrossEntropyLoss : public Loss {
 public:
  float compute_loss() override;
  void backward() override;
};

}  // namespace mlengine::core