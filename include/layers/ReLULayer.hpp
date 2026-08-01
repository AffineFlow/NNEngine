#pragma once
#include "core/Layer.hpp"

namespace affineflow::layers {

/**
 * @brief Elementwise rectified linear activation.
 */
class ReLULayer : public core::Layer {
 public:
  autograd::Tensor* forward(autograd::Tensor* input) override;
};

}  // namespace affineflow::layers