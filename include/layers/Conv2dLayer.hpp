#pragma once

#include <map>
#include <string>
#include <vector>

#include "autograd/Tensor.hpp"
#include "core/Layer.hpp"

namespace affineengine::layers {

/**
 * @brief Applies a 2D convolution over an input signal.
 */
class Conv2dLayer : public core::Layer {
  autograd::Tensor w_;
  autograd::Tensor bias_;
  int in_channels_, out_channels_, in_h_, in_w_, kernel_size_, stride_, pad_;

 public:
  /**
   * @brief Construct a 2D Convolutional Layer.
   */
  Conv2dLayer(int in_channels, int out_channels, int in_h, int in_w,
              int kernel_size, int stride = 1, int pad = 0);

  autograd::Tensor* forward(autograd::Tensor* input) override;
  std::vector<autograd::Tensor*> parameters() override;
  std::map<std::string, autograd::Tensor*> named_parameters() override;
};

}  // namespace affineengine::layers