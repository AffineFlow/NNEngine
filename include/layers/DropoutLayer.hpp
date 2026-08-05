#pragma once
#include "core/Layer.hpp"

namespace affineflow::nn::layers {

/**
 * @brief Randomly zeroes some of the elements of the input tensor.
 */
class DropoutLayer : public core::Layer {
 public:
  float p_;

  /**
   * @brief Create a Dropout layer.
   * @param p Probability of an element to be zeroed. Default: 0.5
   */
  explicit DropoutLayer(float p = 0.5f);

  autograd::Tensor* forward(autograd::Tensor* input) override;
};

}  // namespace affineflow::nn::layers