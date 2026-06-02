#pragma once
#include "core/Layer.hpp"

namespace mlengine::layers {

/**
 * @brief Elementwise rectified linear activation.
 */
class ReLULayer : public core::Layer {
 public:
  autograd::Tensor* forward(autograd::Tensor* input) override;
};

}  // namespace mlengine::layers