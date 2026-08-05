#pragma once
#include "core/Layer.hpp"

namespace affineflow::nn::layers {

class MaxPool2dLayer : public core::Layer {
  int channels_, in_h_, in_w_, kernel_size_, stride_, pad_;

 public:
  MaxPool2dLayer(int channels, int in_h, int in_w, int kernel_size,
                 int stride = 2, int pad = 0);
  autograd::Tensor* forward(autograd::Tensor* input) override;
};

class AvgPool2dLayer : public core::Layer {
  int channels_, in_h_, in_w_, kernel_size_, stride_, pad_;

 public:
  AvgPool2dLayer(int channels, int in_h, int in_w, int kernel_size,
                 int stride = 2, int pad = 0);
  autograd::Tensor* forward(autograd::Tensor* input) override;
};

}  // namespace affineflow::nn::layers