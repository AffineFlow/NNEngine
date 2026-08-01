#pragma once
#include "core/Layer.hpp"

namespace affineflow::layers {

/**
 * @brief Elementwise leaky rectified linear activation.
 */
class LeakyReLULayer : public core::Layer {
 private:
  float alpha_;

 public:
  /**
   * @brief Construct a leaky ReLU layer with the given negative slope.
   * @param alpha Slope applied to negative activations.
   */
  explicit LeakyReLULayer(float alpha = 0.01f);

  autograd::Tensor* forward(autograd::Tensor* input) override;
};

}  // namespace affineflow::layers